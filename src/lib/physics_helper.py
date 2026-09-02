import numpy as np


def calculate_momenta(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    particle_momenta = masses[:, None] * velocities

    linear_momentum = particle_momenta.sum(axis=0)

    center_of_mass = (masses[:, None] * positions).sum(axis=0) / masses.sum()

    relative_positions = positions - center_of_mass

    angular_momentum = np.cross(
        relative_positions,
        particle_momenta,
    ).sum(axis=0)

    return linear_momentum, angular_momentum
