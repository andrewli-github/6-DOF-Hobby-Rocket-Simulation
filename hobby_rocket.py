# Andrew Li
# 7/31/2026
### Simulating projectile motion with 2D newtonian gravity

from time import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

### Create Planet class
class Planet():
    def __init__(self):
        self.radius = 6.371e6 #m
        self.mass = 5.972e24 #kg
        self.G = 6.67430e-11 #m^3/kg/s^2\
        self.mu = self.G * self.mass # m^3/s^2
    
    def gravity(self, x, z):
        r = np.sqrt(x**2 + z**2)  # distance from the center of the planet
        accel_x = self.mu / (r**3)*x
        accel_z = self.mu / (r**3)*z
        return np.asarray([accel_x, accel_z]),r
    
    def altitude(self, x, z):
        r = np.sqrt(x**2 + z**2)
        return r - self.radius

### Create Rocket class
class Rocket():
    def __init__(self, mass, max_thrust, Isp, cross_section, t_MECO, drag_coefficient):
        self.mass = mass
        self.max_thrust = max_thrust
        self.Isp = Isp
        self.v_exit = Isp * 9.81 # m/s
        self.cross_section = cross_section # m^2 
        self.drag_coefficient = drag_coefficient # dimensionless (typical for a rocket)
        self.t_MECO = t_MECO # Time of Main Engine Cut Off (s)

    ### Propulsion model (off of time but make it closed loop in the future with certain altitudes)
    def propulsion(self, t):
        # Thurst timing
        if t < self.t_MECO:
            theta = np.deg2rad(0.0) # angle of thrust in degrees
            thrust_force = self.max_thrust
            m_dot = -thrust_force/self.v_exit
        if t >= self.t_MECO:
            theta = 0.0
            thrust_force = 0.0
            m_dot = 0.0
        thrust_x = thrust_force * np.cos(theta)
        thrust_z = thrust_force * np.sin(theta)
        
        return np.asarray([thrust_x, thrust_z]),m_dot

### Create Aerodynamic class
class Aero():
    def __init__(self):
        self.beta = 0.1354/1000.0 # density constant
        self.rhos = 1.225 # sea level density kg/m^3

    def getDensity(self, altitude):
        return self.rhos * np.exp(-self.beta * altitude)

class AirBrake():
    def __init__(self, rocket):
        self.rocket = rocket
        self.brake_area = 5.0 # m^2
        self.total_area = rocket.cross_section + self.brake_area

    def set_brake_area(self, percent):
        self.calculate_brake_area = self.brake_area * percent
        self.total_area = self.rocket.cross_section + self.calculate_brake_area
        return self.total_area

class AltitudeController():
    def __init__(self, target_altitude, planet):
        self.target_altitude = target_altitude
        self.planet = planet
        self.Kp = 1/800

    def predict_altitude(self, state):
        current_position = np.asarray([state[0], state[1]])
        velocity = np.asarray([state[2], state[3]])
        rad_vel_dir = current_position / np.linalg.norm(current_position)
        rad_vel = np.dot(velocity, rad_vel_dir)
        rad_vel = max(rad_vel, 0.0)  # Ensure non-negative radial velocity

        gravity = self.planet.gravity(state[0], state[1])[0]

        predicted_altitude = self.planet.altitude(state[0], state[1]) + rad_vel**2 / (2 * np.linalg.norm(gravity))
        return predicted_altitude

    def find_error(self, state):
        error = self.predict_altitude(state) - self.target_altitude
        return error
    
    def control_airbrake(self, state, time, t_MECO):
        if time < t_MECO:
            return 0.0  # No braking during powered flight
        else:
            overshoot = self.find_error(state)
            if overshoot < 0.0: #undershoot
                return 0.0
            else: #overshoot
                if overshoot < 10.0: # deadband
                    return 0.0
                else:
                    brake_percent = np.clip(overshoot*self.Kp, 0.0, 1.0)
            return brake_percent
                
# Create aerodynamic model
aero_model = Aero()

# Create planet model
planet_model = Planet()

# Create rocket model
rocket_1 = Rocket(mass=500.0, max_thrust=10000.0, Isp=220.0, cross_section=1.0, t_MECO = 15.5,drag_coefficient=0.4)

air_brake = AirBrake(rocket_1)

altitude_controller = AltitudeController(target_altitude=1650.0, planet=planet_model)
    
class Simulation():
    def __init__(self, rocket, planet, aero, air_brake, altitude_controller):
        self.rocket = rocket
        self.planet = planet
        self.aero = aero
        self.air_brake = air_brake
        self.altitude_controller = altitude_controller

    ### Differential equations
    def derivatives(self, state, t):
        # state vector
        x = state[0]  # altitude from center of planet along north pole
        z = state[1]  # altidude from center of planet along equator
        velx = state[2]  # velocity
        velz = state[3]  # velocity
        mass = state[4]  # mass


        # Kinematic relationship
        x_dot = velx
        z_dot = velz

        # Total Force
        # Gravity
        gravity_accel, r = self.planet.gravity(x, z)
        force_gravity = -gravity_accel*mass

        # Aerodynamic drag
        altitude = self.planet.altitude(x, z)
        air_density = self.aero.getDensity(altitude)
        v_mag = np.sqrt(velx**2 + velz**2) # Calculate velocity magnitude

        # Find brake area
        brake_area_percent = self.altitude_controller.control_airbrake(state, t, self.rocket.t_MECO)
        total_area = self.air_brake.set_brake_area(brake_area_percent)
        
        # Apply F_drag = -0.5 * rho * Cd * A * |v| * v_vector
        force_drag = -0.5 * air_density * self.rocket.drag_coefficient * total_area * v_mag * np.asarray([velx, velz])

        # Thrust force
        force_thrust, m_dot = self.rocket.propulsion(t)

        Total_Force = force_gravity + force_drag + force_thrust

        # Acceleration
        if mass > 0:
            ddot = Total_Force/mass
        else:
            ddot = np.asarray([0.0, 0.0])
            m_dot = 0.0

        # Compute z' 
        if self.ground_check(state, t):
            state_dot = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            state_dot = np.asarray([x_dot, z_dot, ddot[0], ddot[1], m_dot])
        return state_dot
        
    def ground_check(self, state, t):
        if self.planet.altitude(state[0], state[1]) < 0.0 and t > 0.0:
            return True
        else:   
            return False
        
    def run(self, state_int, tspan):
        # Integration call
        self.tspan = tspan
        
        self.state_out = sci.odeint(self.derivatives, state_int, tspan)
        return self.state_out

# Initial conditions for rokcet
x_int = planet_model.radius
z_int = 0.0
velx_int = 0.0 #m/s
velz_int = 0.0 #m/s
period = 100.0 #s (Increased to 100 minutes so you can see the full orbit)
state_int = np.asarray([x_int, z_int, velx_int, velz_int, rocket_1.mass])

class PlotSimulation:
    def __init__(self, simulation):
        self.simulation = simulation

    def plot_results(self):
        state_out = self.simulation.state_out

        self.plot_altitude(state_out)
        self.plot_velocity(state_out)
        self.plot_mass(state_out)
        self.plot_aerodynamic_density()
        self.plot_air_brake_area(state_out)

    def plot_altitude(self, state_out):
        x_out = state_out[:, 0]
        z_out = state_out[:, 1]

        altitude = (
            np.sqrt(x_out**2 + z_out**2)
            - self.simulation.planet.radius
        )

        plt.figure()
        plt.plot(self.simulation.tspan, altitude)
        plt.xlabel("Time (s)")
        plt.ylabel("Altitude (m)")
        plt.title("Rocket Altitude")
        plt.grid(True)
        plt.show()

        # print max altitude
        max_altitude = np.max(altitude)
        print(f"Max Altitude: {max_altitude:.2f} m")

    def plot_velocity(self, state_out):
        velx_out = state_out[:, 2]
        velz_out = state_out[:, 3]
        velocity = np.sqrt(velx_out**2 + velz_out**2)

        plt.figure()
        plt.plot(self.simulation.tspan, velocity)
        plt.xlabel("Time (s)")
        plt.ylabel("Velocity (m/s)")
        plt.title("Rocket Speed")
        plt.grid(True)
        plt.show()

    def plot_mass(self, state_out):
        mass_out = state_out[:, 4]

        plt.figure()
        plt.plot(self.simulation.tspan, mass_out)
        plt.xlabel("Time (s)")
        plt.ylabel("Mass (kg)")
        plt.title("Rocket Mass Over Time")
        plt.grid(True)
        plt.show()

    def plot_aerodynamic_density(self):
        test_altitude = np.linspace(0, 100000, 100)
        test_rho = self.simulation.aero.getDensity(test_altitude)

        plt.figure()
        plt.plot(test_altitude, test_rho)
        plt.xlabel("Altitude (m)")
        plt.ylabel("Air Density (kg/m³)")
        plt.title("Air Density vs. Altitude")
        plt.grid(True)
        plt.show()
    
    def plot_air_brake_area(self, state_out):
        x_out = state_out[:, 0]
        z_out = state_out[:, 1]

        altitude = (
            np.sqrt(x_out**2 + z_out**2)
            - self.simulation.planet.radius
        )

        brake_area = np.where(
            (self.simulation.tspan >= self.simulation.rocket.t_MECO)
            & (altitude >= 1000.0),
            0.5,
            0.0
        )

        plt.figure()
        plt.step(
            self.simulation.tspan,
            brake_area,
            where="post"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Brake Area (m²)")
        plt.title("Air-Brake Deployment")
        plt.ylim(-0.05, 0.55)
        plt.grid(True)
        plt.show()

# Time window
tspan = np.linspace(0, period, 1000)

# Integration call
simulation = Simulation(rocket_1, planet_model, aero_model, air_brake, altitude_controller)
state_out = simulation.run(state_int, tspan)

plot = PlotSimulation(simulation)
plot.plot_results()
