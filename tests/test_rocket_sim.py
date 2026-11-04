"""Physics and regression checks: python -m unittest discover -s tests -v."""
from dataclasses import replace
import unittest

import numpy as np
from numpy.testing import assert_allclose

from rocket_sim import (G, Rocket, aerodynamic_loads, command_brakes,
                        derivatives, predict_apogee, rotation, simulate)


class FlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rocket = Rocket()
        cls.flight = simulate(cls.rocket)
        cls.baseline = simulate(cls.rocket, airbrakes=False)

    def test_flight_invariants_and_burnout(self):
        f, r = self.flight, self.rocket
        self.assertTrue(np.isfinite(f.state).all())
        self.assertTrue((np.diff(f.time) > 0).all())
        assert_allclose(np.linalg.norm(f.state[:, 6:10], axis=1), 1., atol=1e-12)
        expected_mass = r.dry_mass + r.propellant_mass*np.maximum(1-f.time/r.burn_time, 0.)
        assert_allclose(f.state[:, 13], expected_mass, atol=1e-8)
        self.assertTrue(((f.state[:, 14] >= 0) & (f.state[:, 14] <= 1)).all())
        self.assertTrue((f.state[f.time <= r.burn_time, 14] == 0).all())
        self.assertLess(abs(f.state[-1, 5]), 1e-7)
        self.assertTrue((f.state[:, 2] >= -1e-9).all())
        before = f.time <= f.rail_exit
        assert_allclose(f.state[before, 6:10],
                        np.tile(f.state[0, 6:10], (sum(before), 1)), atol=1e-10)
        self.assertGreater(np.max(np.abs(f.state[:, 10:13])), .001)

    def test_airbrakes_reduce_apogee(self):
        self.assertGreater(self.baseline.apogee - self.flight.apogee, 75.)
        self.assertLess(abs(self.flight.apogee - self.rocket.target_apogee), 3.)
        self.assertEqual(self.baseline.state[:, 14].max(), 0.)
        rates = np.diff(self.flight.state[:, 14])/np.diff(self.flight.time)
        self.assertLessEqual(abs(rates).max(), self.rocket.actuator_rate + 1e-7)

    def test_timestep_convergence(self):
        finer = simulate(self.rocket, max_step=.01)
        self.assertLess(abs(finer.apogee - self.flight.apogee), .05)
        assert_allclose(finer.state[-1, :3], self.flight.state[-1, :3], atol=.05)

    def test_vertical_vacuum_coast_conserves_energy(self):
        r = replace(self.rocket, cd=0., normal_slope=0., damping=0.,
                    tilt_deg=0., wind=(0., 0., 0.))
        f = simulate(r, airbrakes=False)
        coast = f.state[f.time >= r.burn_time]
        energy = .5*np.sum(coast[:, 3:6]**2, axis=1) + G*coast[:, 2]
        assert_allclose(energy, energy[0], rtol=1e-8)
        assert_allclose(f.state[:, [0, 1, 3, 4, 10, 11, 12]], 0., atol=1e-10)
        expected_speed = (r.thrust/(r.propellant_mass/r.burn_time)
                          * np.log((r.dry_mass+r.propellant_mass)/r.dry_mass)
                          - G*r.burn_time)
        burnout = np.argmin(abs(f.time-r.burn_time))
        self.assertAlmostEqual(f.state[burnout, 5], expected_speed, places=6)

    def test_unreachable_target_and_invalid_inputs(self):
        high = replace(self.rocket, target_apogee=10000.)
        for y in self.baseline.state:
            self.assertEqual(command_brakes(high, y, high.burn_time), 0.)
        with self.assertRaises(ValueError):
            Rocket(dry_mass=0.)
        with self.assertRaises(ValueError):
            Rocket(wind=(0., float('nan'), 0.))
        with self.assertRaises(ValueError):
            simulate(replace(self.rocket, thrust=1.))
        with self.assertRaises(ValueError):
            simulate(max_step=0.)
        with self.assertRaises(RuntimeError):
            simulate(max_time=.1)


class LoadTests(unittest.TestCase):
    def setUp(self):
        self.r = Rocket(wind=(0., 0., 0.))
        self.y = np.zeros(15)
        self.y[6] = 1.
        self.y[13] = self.r.dry_mass

    def test_drag_dissipates_energy_and_brakes_add_drag(self):
        self.y[3:6] = [4., 2., 100.]
        force, torque = aerodynamic_loads(self.r, self.y)
        self.assertLess(np.dot(force, self.y[3:6]), 0.)
        self.assertGreater(torque[1], 0.)  # nose rotates into +east velocity
        self.assertLess(torque[0], 0.)
        self.y[14] = 1.
        braked, _ = aerodynamic_loads(self.r, self.y)
        self.assertLess(np.dot(braked, self.y[3:6]), np.dot(force, self.y[3:6]))

    def test_quaternion_kinematics_and_euler_rotation(self):
        self.y[10:13] = [.2, -.3, .4]
        r = replace(self.r, damping=0.)
        d = derivatives(3., self.y, r, 0., False, False)
        assert_allclose(d[6:10], [0., .1, -.15, .2])
        inertia = r.inertia(self.y[13])
        assert_allclose(d[10:13], -np.cross(self.y[10:13], inertia*self.y[10:13])/inertia)
        q = np.array([np.cos(.2), 0., np.sin(.2), 0.])
        matrix = rotation(q)
        assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(matrix), 1.)

    def test_predictor_vacuum_limit_and_controller_saturation(self):
        self.y[2], self.y[5] = 100., 80.
        vacuum = replace(self.r, cd=0., brake_area=0.)
        self.assertAlmostEqual(predict_apogee(vacuum, self.y, 0.), 100.+80.**2/(2*G))
        low = replace(self.r, target_apogee=1.)
        self.assertEqual(command_brakes(low, self.y, 0.), 0.)
        self.assertEqual(command_brakes(low, self.y, low.burn_time), 1.)
        self.y[5] = -1.
        self.assertEqual(command_brakes(low, self.y, 10.), 0.)


if __name__ == '__main__':
    unittest.main()
