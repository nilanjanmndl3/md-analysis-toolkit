"""Run the toolkit on a reproducible synthetic random-walk trajectory."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running directly from a source checkout without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdtoolkit import (  # noqa: E402
    mean_squared_displacement,
    diffusion_coefficient,
    radial_distribution_function,
    density_profile,
)


def main():
    rng = np.random.default_rng(17)
    n_frames, n_particles = 500, 250
    dt = 0.01
    box_length = 20.0
    step_sigma = 0.12

    # Independent Brownian random walks; unwrapped coordinates for MSD.
    increments = rng.normal(0.0, step_sigma, size=(n_frames - 1, n_particles, 3))
    positions = np.zeros((n_frames, n_particles, 3), dtype=float)
    positions[1:] = np.cumsum(increments, axis=0)

    times, msd = mean_squared_displacement(positions, dt)
    d_est, slope = diffusion_coefficient(times, msd, fit_start=0.5, fit_end=3.0)

    # RDF and density profile use coordinates wrapped into a cubic demonstration box.
    wrapped = np.mod(positions, box_length)
    radii, gr = radial_distribution_function(wrapped, box_length, bins=60)
    z, rho = density_profile(wrapped, box_length, bins=40, axis=2)

    out = ROOT / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(times, msd, label="MSD")
    ax.set(xlabel="Time", ylabel="MSD", title="Mean squared displacement")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "msd.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(radii, gr)
    ax.set(xlabel="r", ylabel="g(r)", title="Radial distribution function")
    fig.tight_layout()
    fig.savefig(out / "rdf.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(z, rho)
    ax.set(xlabel="z", ylabel="Number density", title="Number-density profile")
    fig.tight_layout()
    fig.savefig(out / "density_profile.png", dpi=180)
    plt.close(fig)

    print(f"Estimated diffusion coefficient: {d_est:.5f} coordinate_units^2/time_units")
    print(f"MSD fit slope: {slope:.5f}")
    print(f"Figures saved in: {out}")


if __name__ == "__main__":
    main()
