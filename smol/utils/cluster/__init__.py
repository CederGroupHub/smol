"""Cython based utilities for computing correlations and cluster interactions."""


def get_orbit_data(orbits):
    """Gather orbit data necessary to create a ClusterSpaceEvaluator."""
    orbit_data = tuple(
        (
            orbit.id,
            orbit.bit_id,
            orbit.flat_correlation_tensors,
            orbit.flat_tensor_indices,
        )
        for orbit in orbits
    )
    return orbit_data
