"""Run a six-DOF hobby rocket ascent with and without apogee-control airbrakes."""
import argparse
import csv
from pathlib import Path

import numpy as np

from rocket_sim import Rocket, simulate


def save_csv(path, result):
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time_s', 'east_m', 'north_m', 'altitude_m',
                         've_mps', 'vn_mps', 'vu_mps', 'qw', 'qx', 'qy', 'qz',
                         'p_radps', 'q_radps', 'r_radps', 'mass_kg', 'brake_fraction',
                         'brake_command', 'predicted_apogee_m'])
        writer.writerows(zip(result.time, *result.state.T, result.command, result.predicted_apogee))


def plot_comparison(path, rocket, flights):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 2, figsize=(11, 10))
    for label, result in flights.items():
        t, y = result.time, result.state
        axes[0, 0].plot(t, y[:, 2], label=label)
        axes[0, 1].plot(t, np.linalg.norm(y[:, 3:6], axis=1), label=label)
        axes[1, 0].plot(t, y[:, 14], label=label)
        axes[1, 1].plot(t, y[:, 13], label=label)
        axes[2, 0].plot(t, np.rad2deg(np.linalg.norm(y[:, 10:13], axis=1)), label=label)
        axes[2, 1].plot(y[:, 0], y[:, 1], label=label)
    axes[0, 0].axhline(rocket.target_apogee, color='k', linestyle=':', label='Target')
    labels = ['Altitude AGL (m)', 'Speed (m/s)', 'Airbrake fraction',
              'Mass (kg)', 'Angular speed (deg/s)', 'North (m)']
    for ax, ylabel in zip(axes.flat, labels):
        ax.set(xlabel='Time (s)', ylabel=ylabel)
        ax.grid(True)
        ax.legend()
    axes[2, 1].set_xlabel('East (m)')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection='3d')
    for label, result in flights.items():
        ax.plot(*result.state[:, :3].T, label=label)
    ax.set(xlabel='East (m)', ylabel='North (m)', zlabel='Altitude AGL (m)')
    ax.legend()
    fig.savefig(path.with_name('trajectory_3d.png'), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=float, default=400., help='Target apogee AGL (m)')
    parser.add_argument('--wind', type=float, nargs=3, default=(2., 0., 0.),
                        metavar=('EAST', 'NORTH', 'UP'), help='Steady wind (m/s)')
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    rocket = Rocket(target_apogee=args.target, wind=tuple(args.wind))
    flights = {'Unbraked': simulate(rocket, airbrakes=False), 'Controlled': simulate(rocket)}
    args.output.mkdir(parents=True, exist_ok=True)
    for label, result in flights.items():
        save_csv(args.output / f'{label.lower()}.csv', result)
        speed = np.linalg.norm(result.state[:, 3:6], axis=1).max()
        print(f'{label}: apogee {result.apogee:.2f} m at {result.time[-1]:.2f} s; '
              f'rail exit {result.rail_exit:.3f} s; peak speed {speed:.2f} m/s')
    controlled = flights['Controlled']
    print(f'Target error: {controlled.apogee - rocket.target_apogee:+.2f} m')
    if abs(controlled.apogee - rocket.target_apogee) > 10:
        print('Target missed: braking authority and the approximate predictor are limited.')
    if not args.no_plots:
        plot_comparison(args.output / 'flight_comparison.png', rocket, flights)


if __name__ == '__main__':
    main()
