"""Six-degree-of-freedom hobby rocket ascent and sampled airbrake control.

SI units; local East/North/Up frame; body +Z points toward the nose.
State: ENU position(3), ENU velocity(3), body-to-ENU quaternion wxyz(4),
body angular velocity(3), remaining vehicle mass(1), brake deployment(1).
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp

G = 9.80665


@dataclass(frozen=True)
class Rocket:
    dry_mass: float = 2.5
    propellant_mass: float = .5
    thrust: float = 150.0
    burn_time: float = 2.5
    length: float = 1.5
    diameter: float = .1
    cd: float = .45
    normal_slope: float = 8.0
    cp_aft: float = .18  # CP aft of fixed CG, metres
    damping: float = .015  # torque = -coefficient * airspeed * body rate
    brake_area: float = .012  # total projected area of symmetric petals, m^2
    brake_cd: float = 1.2
    actuator_tau: float = .15
    actuator_rate: float = 2.0  # deployment fraction per second
    target_apogee: float = 400.0
    rail_length: float = 2.0
    tilt_deg: float = 3.0
    azimuth_deg: float = 0.0  # from east toward north
    wind: tuple = (2.0, 0.0, 0.0)

    def __post_init__(self):
        positive = ('dry_mass', 'propellant_mass', 'thrust', 'burn_time',
                    'length', 'diameter', 'actuator_tau', 'actuator_rate',
                    'target_apogee', 'rail_length')
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('cd', 'normal_slope', 'cp_aft', 'damping', 'brake_area', 'brake_cd'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not math.isfinite(self.tilt_deg) or not 0 <= self.tilt_deg < 45:
            raise ValueError('tilt_deg must be in [0, 45)')
        if (not math.isfinite(self.azimuth_deg) or np.shape(self.wind) != (3,)
                or not np.isfinite(self.wind).all()):
            raise ValueError('azimuth and three-component wind must be finite')

    @property
    def area(self):
        return math.pi * self.diameter**2 / 4

    def inertia(self, mass):
        # Uniform-cylinder approximation about a fixed CG, diagonal in body axes.
        radius = self.diameter / 2
        return mass * np.array([(3*radius**2 + self.length**2)/12]*2 + [radius**2/2])


def rotation(quaternion):
    """Body-to-earth rotation matrix, scalar-first quaternion."""
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def density(height):
    return 1.225 * math.exp(-max(height, 0.0) / 8500.0)


def aerodynamic_loads(rocket, state):
    """Return aerodynamic force in ENU and moment about CG in body axes."""
    matrix = rotation(state[6:10])
    velocity = matrix.T @ (state[3:6] - np.asarray(rocket.wind))
    speed = np.linalg.norm(velocity)
    if speed < 1e-9:
        return np.zeros(3), np.zeros(3)
    pressure = .5 * density(state[2]) * speed**2
    cda = (rocket.cd*rocket.area
           + rocket.brake_cd*rocket.brake_area*np.clip(state[14], 0, 1))
    drag = -pressure * cda * velocity / speed
    # Bounded crossflow model: approximately linear in incidence at small angles.
    normal = (-pressure * rocket.area * rocket.normal_slope
              * np.array([velocity[0], velocity[1], 0.0]) / speed)
    torque = np.cross([0.0, 0.0, -rocket.cp_aft], normal)
    torque -= rocket.damping * speed * state[10:13]
    # Perfectly symmetric brakes add drag but no net control torque.
    return matrix @ (drag + normal), torque


def predict_apogee(rocket, state, deployment):
    """Vertical coast predictor with frozen density and drag area.

    Integrate dv/dt = -g-k*v^2 analytically. This lightweight controller model
    does not replace the full six-DOF plant, and is approximate in wind/tilt.
    """
    vz = max(state[5], 0.0)
    cda = rocket.cd*rocket.area + rocket.brake_cd*rocket.brake_area*deployment
    k = .5 * density(state[2]) * cda / state[13]
    rise = vz*vz/(2*G) if k < 1e-12 else math.log1p(k*vz*vz/G)/(2*k)
    return state[2] + rise


def command_brakes(rocket, state, time):
    """Bisection finds the coast deployment needed to approach target apogee."""
    if time < rocket.burn_time or state[5] <= 0:
        return 0.0
    if predict_apogee(rocket, state, 0.0) <= rocket.target_apogee:
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(20):
        middle = (low + high)/2
        if predict_apogee(rocket, state, middle) > rocket.target_apogee:
            low = middle
        else:
            high = middle
    return high


def derivatives(time, state, rocket, command, on_rail, powered):
    matrix = rotation(state[6:10])
    mass = max(state[13], rocket.dry_mass)
    thrust = rocket.thrust if powered else 0.0
    mdot = -rocket.propellant_mass/rocket.burn_time if powered else 0.0
    aero, torque = aerodynamic_loads(rocket, state)
    acceleration = ((matrix @ np.array([0.0, 0.0, thrust]) + aero)/mass
                    - np.array([0.0, 0.0, G]))
    omega = state[10:13]
    w, vector = state[6], state[7:10]
    qdot = .5 * np.r_[-np.dot(vector, omega), w*omega + np.cross(vector, omega)]
    inertia = rocket.inertia(mass)
    # Symmetric exhaust carries away the lost mass's angular momentum.
    # No separate -I_dot*omega term is imposed on the remaining body.
    omega_dot = (torque - np.cross(omega, inertia*omega))/inertia
    velocity = state[3:6]
    if on_rail:
        axis = matrix[:, 2]
        along = np.dot(acceleration, axis)
        if np.dot(state[:3], axis) <= 0 and np.dot(velocity, axis) <= 0:
            along = max(along, 0.0)  # launch-pad reaction
        acceleration = axis*along
        velocity = axis*max(np.dot(velocity, axis), 0.0)
        qdot = np.zeros(4)
        omega_dot = np.zeros(3)
    brake_dot = np.clip((command-state[14])/rocket.actuator_tau,
                        -rocket.actuator_rate, rocket.actuator_rate)
    return np.r_[velocity, acceleration, qdot, omega_dot, mdot, brake_dot]


@dataclass
class Flight:
    time: np.ndarray
    state: np.ndarray
    command: np.ndarray  # command held over the following interval
    predicted_apogee: np.ndarray
    rail_exit: float

    @property
    def apogee(self):
        return float(self.state[-1, 2])


def simulate(rocket=None, airbrakes=True, control_period=.1, max_step=.02, max_time=60.0):
    """Integrate ignition to first apogee; raise on failed launch or timeout.

    Controller commands are sampled and held. Each burn/rail boundary is an
    exact integration boundary; apogee is located by a terminal root event.
    """
    rocket = rocket or Rocket()
    if not all(np.isfinite(v) and v > 0 for v in (control_period, max_step, max_time)):
        raise ValueError('time settings must be finite and positive')
    tilt, azimuth = np.deg2rad([rocket.tilt_deg, rocket.azimuth_deg])
    axis = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    state = np.zeros(15)
    state[6:10] = np.r_[math.cos(tilt/2), axis*math.sin(tilt/2)]
    state[13] = rocket.dry_mass + rocket.propellant_mass
    rail_axis = rotation(state[6:10])[:, 2]
    if rocket.thrust*np.cos(tilt) <= state[13]*G:
        raise ValueError('motor cannot lift this rocket from the pad')
    t, on_rail, rail_exit = 0.0, True, math.nan
    times, states, commands = [t], [state.copy()], []

    def rail_event(t, y):
        return np.dot(y[:3], rail_axis) - rocket.rail_length

    def apogee_event(t, y):
        return y[5]

    rail_event.terminal, rail_event.direction = True, 1
    apogee_event.terminal, apogee_event.direction = True, -1
    while t < max_time - 1e-10:
        powered = t < rocket.burn_time - 1e-9
        end = min(t+control_period, max_time)
        if powered:
            end = min(end, rocket.burn_time)
        command = command_brakes(rocket, state, t) if airbrakes and not on_rail else 0.0
        commands.append(command)
        result = solve_ivp(lambda t, y: derivatives(t, y, rocket, command, on_rail, powered),
                           (t, end), state, events=rail_event if on_rail else apogee_event,
                           max_step=max_step, rtol=1e-7, atol=1e-9)
        if not result.success:
            raise RuntimeError(result.message)
        t, state = result.t[-1], result.y[:, -1].copy()
        state[6:10] /= np.linalg.norm(state[6:10])
        state[13] = max(rocket.dry_mass, state[13])
        state[14] = np.clip(state[14], 0.0, 1.0)
        times.append(t)
        states.append(state.copy())
        if result.t_events[0].size:
            if on_rail:
                rail_exit, on_rail = t, False
                if t >= rocket.burn_time and state[5] <= 0:
                    raise RuntimeError('rocket stalled on launch rail')
            else:
                states = np.array(states)
                return Flight(np.array(times), states, np.r_[commands, 0.0],
                              np.array([predict_apogee(rocket, y, y[14]) for y in states]),
                              rail_exit)
    raise RuntimeError('apogee not reached before max_time; check parameters or increase max_time')
