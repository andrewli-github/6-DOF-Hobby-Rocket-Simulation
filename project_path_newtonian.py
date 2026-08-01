# Andrew Li
# 6/31/2026
### Simulating projectile motion with 1D newtonian gravity

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

# Constant parameters
mass = 1.0 #kg
G = 6.67430e-11 #m^3/kg/s^2

# Planet parameters
Radius_planet = 6.371e6 #m
Mass_planet = 5.972e24 #kg

### Gravitational acceleration as a function of altitude
def gravity(z):
    r = np.sqrt(z**2)  # distance from the center of the planet
    if r < Radius_planet:
        acceleration = 0.0
    else:
        acceleration = G * Mass_planet / (r**3)*r
    return acceleration

### Differential equations
def Derivatives(state, t):
    # state vector
    z = state[0]  # position
    velz = state[1]  # velocity

    # Kinematic relationship
    z_dot = velz

    # Total Force
    # Gravity
    force_gravity = -gravity(z)*mass

    # Aerodynamic drag
    force_drag = 0.0

    # Thrust force
    force_thrust = 0.0

    Total_Force = force_gravity + force_drag + force_thrust

    # Acceleration
    z_ddot = Total_Force/mass

    # After landing ground state
    if z <= Radius_planet and z_dot < 0:
        z_ddot = 0
        z_dot = 0

    # Compute z' 
    state_dot = np.asarray([z_dot, z_ddot])
    return state_dot

# Initial conditions
z_int = Radius_planet
velz_int = 2000 #m/s
state_int = np.asarray([z_int, velz_int])

# Time window
tspan = np.linspace(0, 500, 1000)

# Integration call
state_out = sci.odeint(Derivatives, state_int, tspan)

# outputs
z_out = state_out[:,0]
altitude = z_out - Radius_planet
velz_out = state_out[:,1]
print(velz_out[:10])

# Plot Altidude
plt.plot(tspan, altitude)
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.grid(True)
plt.show()

# PLot Velocity
plt.plot(tspan, velz_out)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.grid(True)
plt.show()