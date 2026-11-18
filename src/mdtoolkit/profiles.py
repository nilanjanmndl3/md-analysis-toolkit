"""Spatial profile analysis."""

import numpy as np


def density_profile(positions, box_length, bins=50, axis=2):
    """Calculate a number-density profile along one Cartesian axis.

    Parameters
    ----------
    positions : array_like, shape (n_frames, n_particles, 3)
        Particle coordinates.
    box_length : float
        Box length along the selected axis.
    bins : int
        Number of spatial bins.
    axis : int
        Cartesian axis: 0=x, 1=y, 2=z.

    Returns
    -------
    centers : ndarray
        Bin-center coordinates.
    number_density : ndarray
        Mean number density in particles per coordinate-unit cubed.
    """
    xyz = np.asarray(positions, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_particles, 3)")
    if box_length <= 0 or bins < 1:
        raise ValueError("box_length and bins must be positive")
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")

    edges = np.linspace(0.0, box_length, bins + 1)
    counts = np.zeros(bins, dtype=float)
    for frame in xyz:
        coord = np.mod(frame[:, axis], box_length)
        hist, _ = np.histogram(coord, bins=edges)
        counts += hist

    bin_width = box_length / bins
    # Assumes a cubic box, so each slab volume is L^2 * bin_width.
    slab_volume = box_length**2 * bin_width
    number_density = counts / (xyz.shape[0] * slab_volume)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, number_density
