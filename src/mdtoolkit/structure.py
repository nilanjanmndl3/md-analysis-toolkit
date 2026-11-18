"""Structural-property analysis."""

import numpy as np


def radial_distribution_function(positions, box_length, bins=100, r_max=None):
    """Calculate a simple RDF for a cubic periodic box.

    Parameters
    ----------
    positions : array_like, shape (n_frames, n_particles, 3)
        Particle coordinates. Coordinates may lie outside the box; they are
        wrapped internally for pair-distance calculation.
    box_length : float
        Cubic box side length, in coordinate units.
    bins : int
        Number of radial bins.
    r_max : float, optional
        Maximum RDF radius. Defaults to half the box length.

    Notes
    -----
    This implementation averages pair counts over frames and normalizes using
    the ideal-gas shell population. It is intended for a simple cubic-box
    demonstration; verify finite-size and cutoff conventions for production use.
    """
    xyz = np.asarray(positions, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_particles, 3)")
    if box_length <= 0 or bins < 1:
        raise ValueError("box_length and bins must be positive")
    n_frames, n_particles, _ = xyz.shape
    if n_particles < 2:
        raise ValueError("At least two particles are required")

    if r_max is None:
        r_max = box_length / 2.0
    if not 0 < r_max <= box_length / 2.0:
        raise ValueError("For this cubic-box implementation, r_max must be <= L/2")

    edges = np.linspace(0.0, r_max, bins + 1)
    counts = np.zeros(bins, dtype=float)
    for frame in xyz:
        wrapped = np.mod(frame, box_length)
        delta = wrapped[:, None, :] - wrapped[None, :, :]
        delta -= box_length * np.rint(delta / box_length)
        distances = np.linalg.norm(delta, axis=2)
        i, j = np.triu_indices(n_particles, k=1)
        hist, _ = np.histogram(distances[i, j], bins=edges)
        counts += 2.0 * hist  # count both members of each pair

    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    number_density = n_particles / box_length**3
    ideal_counts = n_frames * n_particles * number_density * shell_volumes
    g_r = np.divide(counts, ideal_counts, out=np.zeros_like(counts), where=ideal_counts > 0)
    radii = 0.5 * (edges[:-1] + edges[1:])
    return radii, g_r
