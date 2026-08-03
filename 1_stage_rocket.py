# Andrew Li
# 6/31/2026
### Simulating projectile motion with 2D newtonian gravity

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

# Planet parameters
Radius_planet = 6.371e6 #m
Mass_planet = 5.972e24 #kg
G = 6.67430e-11 #m^3/kg/s^2

# Rocket parameters
mass_int = 3500.0 #kg
max_thrust = 180000.0 #N
Isp = 245.0 #s
t_MECO = 32.0 # Main engine cutoff time
t_stage_1 = 2.0 # Stage 1 separation time
mass_stage_1 = 1000.0 #kg
v_exit = Isp*9.81 # m/s

# Initial conditions for rokcet
x_int = Radius_planet
z_int = 0.0
velx_int = 0.0 #m/s
velz_int = 0.0 #m/s
period = 500.0 #s
state_int = np.asarray([x_int, z_int, velx_int, velz_int, mass_int])

### Gravitational acceleration as a function of altitude
def gravity(x, z):
    global Radius_planet, Mass_planet, G
    r = np.sqrt(x**2 + z**2)  # distance from the center of the planet
    if r < Radius_planet:
        accel_x = 0.0
        accel_z = 0.0
    else:
        accel_x = G * Mass_planet / (r**3)*x
        accel_z = G * Mass_planet / (r**3)*z
    return np.asarray([accel_x, accel_z])

### Propulsion model (off of time but make it closed loop in the future with certain altitudes)
def propulsion(t):
    global max_thrust, Isp, t_MECO

    # Thurst timing
    if t < t_MECO:
        thrust_force = max_thrust
        m_dot = -thrust_force/v_exit
    if t >= t_MECO and t < (t_MECO + t_stage_1):
        thrust_force = 0.0

        ## mass loss from stage 1 separation
        m_dot = -mass_stage_1/t_stage_1
    if t >= (t_MECO + t_stage_1):
        thrust_force = 0.0
        m_dot = 0.0

    theta = np.deg2rad(10.0) # angle of thrust in degrees
    thrust_x = thrust_force * np.cos(theta)
    thrust_z = thrust_force * np.sin(theta)
    
    return np.asarray([thrust_x, thrust_z]),m_dot

### Differential equations
def Derivatives(state, t):
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
    gravity_accel = gravity(x, z)
    force_gravity = -gravity_accel*mass

    # Aerodynamic drag
    force_drag = np.asarray([0.0, 0.0])

    # Thrust force
    force_thrust, m_dot = propulsion(t)

    Total_Force = force_gravity + force_drag + force_thrust

    # Acceleration
    if mass > 0:
        ddot = Total_Force/mass
    else:
        ddot = np.asarray([0.0, 0.0])
        m_dot = 0.0

    # Compute z' 
    state_dot = np.asarray([x_dot, z_dot, ddot[0], ddot[1], m_dot])
    return state_dot

# Initial conditions in orbit
"""
x_int = Radius_planet+1200000
z_int = 0.0
r_int = np.sqrt(x_int**2 + z_int**2)
velx_int = 0.0 #m/s
velz_int = np.sqrt(G*Mass_planet/x_int)*1.1 #m/s
state_int = np.asarray([x_int, z_int, velx_int, velz_int])
period = 2*np.pi*np.sqrt(r_int**3/(G*Mass_planet))
"""

# Time window
tspan = np.linspace(0, period, 1000)

# Integration call
state_out = sci.odeint(Derivatives, state_int, tspan)

# outputs
x_out = state_out[:,0]
z_out = state_out[:,1]
altitude = np.sqrt(x_out**2 + z_out**2) - Radius_planet
velx_out = state_out[:,2]
velz_out = state_out[:,3]
mass_out = state_out[:,4]
vel_out = np.sqrt(velx_out**2 + velz_out**2)

## 2d Orbit
plt.plot(x_out, z_out, 'r-', label='Projectile Path')
plt.plot(x_out[0], z_out[0], 'g*')
theta = np.linspace(0, 2*np.pi, 1000)
x_planet = Radius_planet * np.sin(theta)
y_planet = Radius_planet * np.cos(theta)
plt.plot(x_planet, y_planet,'b-', label='Planet')
plt.grid()
plt.legend(loc=1)
plt.show()

# Plot Altidude
plt.figure()
plt.plot(tspan, altitude)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.grid(True)
plt.show()

# PLot Velocity
plt.figure()
plt.plot(tspan, vel_out)
plt.xlabel('Time (s)')
plt.ylabel('Total Speed (m/s)')
plt.grid(True)
plt.show()

# Plot Mass
plt.figure()
plt.plot(tspan, mass_out)
plt.xlabel('Time (s)')
plt.ylabel('Mass (kg)')
plt.grid(True)
plt.show()
