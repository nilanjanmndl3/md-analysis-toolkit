import numpy as np
import pytest

from mdtoolkit.structure import radial_distribution_function
from mdtoolkit.profiles import density_profile


def test_rdf_returns_expected_array_shapes():
    rng = np.random.default_rng(4)
    positions = rng.uniform(0, 10, size=(3, 100, 3))
    r, gr = radial_distribution_function(positions, box_length=10, bins=20)
    assert r.shape == (20,)
    assert gr.shape == (20,)
    assert np.all(gr >= 0)


def test_density_profile_integrates_to_mean_number_density():
    rng = np.random.default_rng(8)
    positions = rng.uniform(0, 10, size=(5, 100, 3))
    z, rho = density_profile(positions, box_length=10, bins=20, axis=2)
    assert z.shape == (20,)
    assert rho.shape == (20,)
    mean_density = rho.mean()
    assert mean_density == pytest.approx(100 / 10**3, rel=1e-12)
