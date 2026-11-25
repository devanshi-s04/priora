"""WP0 smoke test: the package imports and the contracts exist."""

import priora


def test_imports():
    assert priora.__version__
    assert priora.Attributable is not None
    assert priora.PopulationDensity is not None


def test_constants():
    from priora import constants

    assert abs(constants.MU_SUN - constants.K_GAUSS**2) < 1e-18
    assert constants.Q_NEO == 1.3
