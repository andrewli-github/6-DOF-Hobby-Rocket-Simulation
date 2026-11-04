# Six-DOF hobby rocket with airbrakes

A Python simulation of a fin-stabilized hobby-scale rocket, from ignition to
first apogee. `hobby_rocket.py` is the runnable example; `rocket_sim.py` contains
the reusable physics and controller. The earlier standalone staging and
projectile examples have been removed.

## Run

Requires Python 3.10 or newer:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python hobby_rocket.py
python hobby_rocket.py --target 425 --wind 3 1 0
python hobby_rocket.py --no-plots
python -m unittest discover -s tests -v
```

The example runs identical unbraked and controlled launches. It prints apogee,
rail-exit time, peak speed, and target error. `output/` receives both flights as
CSV, a six-panel comparison, and a 3D trajectory image. Plotting uses a headless
backend and does not open blocking windows. Importing either module does not
start a simulation.

Configure the `Rocket` dataclass in `rocket_sim.py` or construct your own:

```python
from rocket_sim import Rocket, simulate
rocket = Rocket(target_apogee=400, wind=(2, 1, 0), tilt_deg=3)
flight = simulate(rocket)
print(flight.apogee)
```

## Example vehicle and coordinates

Defaults are illustrative, not measured motor or airframe data: 2.5 kg dry mass,
0.5 kg propellant, 1.5 m length, 0.10 m diameter, and 150 N constant thrust for
2.5 s. Propellant burns uniformly. Body drag coefficient is 0.45; the combined
fully deployed airbrake area is 0.012 m² with drag coefficient 1.2. A 2 m launch
rail tilts 3 degrees east, and steady wind defaults to 2 m/s east. The target is
400 m above the launch site. Every configurable quantity uses SI units except
explicitly named degree angles.

Earth coordinates are East, North, Up, with the launch point at the origin.
Body +Z points toward the nose; +X/+Y are transverse right-handed axes. The
scalar-first unit quaternion rotates body vectors into earth coordinates.
The 15 states are position (3), earth velocity (3), quaternion (4), body angular
rate (3), mass (1), and actual airbrake fraction (1). A quaternion has four
numbers but only three independent attitude coordinates, so this is six DOF.

## Physics and control

- Translational Newton equations include body-axis thrust rotated into earth
  coordinates, gravity, and aerodynamic forces based on velocity relative to wind.
- Rotational Euler equations include the gyroscopic cross product, aerodynamic
  restoring moments from CP 0.18 m aft of CG, and rate damping. Quaternion
  kinematics avoid Euler-angle singularities. Inertia is approximated by a
  uniform cylinder and changes with mass. Symmetric exhaust is assumed to carry
  away the angular momentum of the expelled mass; CG stays fixed.
- Density decreases exponentially with an 8.5 km scale height. Gravity is constant
  at 9.80665 m/s². Drag opposes relative velocity; bounded transverse normal force
  approximates a small-angle normal-force slope. Symmetric brakes add drag without
  a net attitude-control moment. Roll is integrated and damped but no fin cant or
  imposed roll torque is included.
- The launch rail constrains translation to its axis and locks attitude until an
  exact rail-exit event. Motor cutoff is an exact integration boundary. Flight
  ends at a root-located zero upward velocity, avoiding ground-contact derivative
  discontinuities. No descent, recovery, parachute, or impact model is included.
- A sampled controller (nominal 10 Hz) predicts coast apogee by analytically
  integrating vertical quadratic drag at the current frozen density. Bisection
  finds a deployment fraction between 0 and 1. Commands are held between samples;
  the actuator has a 0.15 s lag and 2 fraction/s rate limit. Braking is inhibited
  during motor burn. Actual deployment, held command, and predicted apogee are
  separate CSV columns. The predictor assumes instantaneous constant deployment;
  actuator lag, wind and rotation are captured by the plant, not the predictor.

The controller can only reduce apogee. A target above the natural apogee leaves
brakes retracted; a low target can saturate the brakes and still be missed.
Default demonstration: approximately 525 m unbraked and 400 m controlled.
The target is not guaranteed for arbitrary vehicle, wind or actuator settings.

## Validation and limits

Tests check drag dissipation and restoring-moment signs, rotation and quaternion
kinematics, mass burnoff, actuator bounds/rate, burnout inhibition, rail attitude,
root-located apogee, a variable-mass analytic burnout speed, vacuum coast energy,
controller saturation, invalid inputs, and integration-step convergence.

This is an educational engineering model. Aerodynamic coefficients, damping,
thrust history, CG and inertia need measured or independently validated inputs
before comparisons with a physical rocket. Constant coefficients and the
crossflow approximation do not capture transonic aerodynamics, large-incidence
flow, structural loads, flexible modes, sensor errors, actuator loading or
asymmetric petal failures. Simulations stop at first apogee; they do not predict
safe recovery or landing.

Background: NASA's [rocket stability conditions](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/conditions-for-rocket-stability/)
explain CP/CG restoring moments, and its [four rocket forces](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/four-rocket-forces/)
describe weight, thrust and aerodynamic forces. The coefficients here are
explicit illustrative assumptions, not NASA-derived vehicle data.
