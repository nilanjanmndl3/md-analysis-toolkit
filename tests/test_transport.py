import numpy as np
import pytest

from mdtoolkit.transport import mean_squared_displacement, diffusion_coefficient


def test_msd_for_constant_displacement():
    positions = np.zeros((4, 2, 3))
    positions[:, :, 0] = np.arange(4)[:, None]
    times, msd = mean_squared_displacement(positions, time_step=0.5)
    np.testing.assert_allclose(times, [0.0, 0.5, 1.0, 1.5])
    np.testing.assert_allclose(msd, [0.0, 1.0, 4.0, 9.0])


def test_diffusion_coefficient_from_known_slope():
    times = np.arange(10, dtype=float)
    msd = 6.0 * 2.5 * times
    diffusion, slope = diffusion_coefficient(times, msd, dimensions=3)
    assert diffusion == pytest.approx(2.5)
    assert slope == pytest.approx(15.0)


def test_msd_rejects_wrong_shape():
    with pytest.raises(ValueError):
        mean_squared_displacement(np.zeros((5, 3)), 1.0)
