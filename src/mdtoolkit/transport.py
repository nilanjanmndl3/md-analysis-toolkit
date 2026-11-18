"""Transport-property analysis for molecular dynamics trajectories."""

import numpy as np


def mean_squared_displacement(positions, time_step):
    """Calculate the MSD relative to the first frame.

    Parameters
    ----------
    positions : array_like, shape (n_frames, n_particles, 3)
        Unwrapped particle coordinates.
    time_step : float
        Time between consecutive frames.

    Returns
    -------
    times : ndarray, shape (n_frames,)
        Time values in the same unit as ``time_step``.
    msd : ndarray, shape (n_frames,)
        Mean squared displacement, in squared coordinate units.
    """
    xyz = np.asarray(positions, dtype=float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, n_particles, 3)")
    if xyz.shape[0] < 2 or xyz.shape[1] < 1:
        raise ValueError("positions must contain at least two frames and one particle")
    if time_step <= 0:
        raise ValueError("time_step must be positive")

    displacement = xyz - xyz[0:1, :, :]
    squared_displacement = np.sum(displacement**2, axis=2)
    msd = np.mean(squared_displacement, axis=1)
    times = np.arange(xyz.shape[0], dtype=float) * time_step
    return times, msd


def diffusion_coefficient(times, msd, fit_start=None, fit_end=None, dimensions=3):
    """Estimate D from a linear fit to the MSD curve.

    The Einstein relation is MSD(t) ~= 2*d*D*t at sufficiently long times,
    where d is the number of spatial dimensions.

    Returns D and the fitted slope in coordinate_units**2 / time_units.
    """
    t = np.asarray(times, dtype=float)
    y = np.asarray(msd, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or t.size != y.size:
        raise ValueError("times and msd must be one-dimensional arrays of equal length")
    if t.size < 2:
        raise ValueError("At least two data points are required")
    if dimensions < 1:
        raise ValueError("dimensions must be a positive integer")

    mask = np.ones(t.size, dtype=bool)
    if fit_start is not None:
        mask &= t >= fit_start
    if fit_end is not None:
        mask &= t <= fit_end
    if np.count_nonzero(mask) < 2:
        raise ValueError("The selected fitting interval must contain at least two points")

    slope, intercept = np.polyfit(t[mask], y[mask], 1)
    diffusion = slope / (2.0 * dimensions)
    return float(diffusion), float(slope)
