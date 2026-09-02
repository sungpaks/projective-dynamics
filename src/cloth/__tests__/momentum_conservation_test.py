import numpy as np
import taichi as ti

from cloth.grid import create_cloth_grid
from cloth.rest_state import create_triangle_rest_state
from cloth.solver import ClothSolver
from lib.physics_helper import calculate_momenta


def test_rest_state_without_external_force_should_preserve_momenta() -> None:
    """외력이 없는 rest 상태에서는 선형·각운동량이 생성되지 않아야 한다."""
    # Given: 변형과 초기 속도가 없고 바닥에서 떨어진 천
    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False)

    initial_positions, triangle_indices, _ = create_cloth_grid(
        4,
        position=(0.0, 2.0, 0.0),
        normal=(0.0, 0.0, 1.0),
    )
    rest_state = create_triangle_rest_state(
        initial_positions,
        triangle_indices,
    )
    solver = ClothSolver(
        initial_positions=initial_positions,
        rest_state=rest_state,
        time_step=1.0 / 60.0,
    )

    masses = np.ones(
        len(initial_positions),
        dtype=np.float64,
    )

    initial_linear, initial_angular = calculate_momenta(
        solver.positions.to_numpy().astype(np.float64),
        solver.velocities.to_numpy().astype(np.float64),
        masses,
    )

    # When: 외력 없이 한 physics step을 진행하면
    solver.step(gravity=(0.0, 0.0, 0.0))

    final_linear, final_angular = calculate_momenta(
        solver.positions.to_numpy().astype(np.float64),
        solver.velocities.to_numpy().astype(np.float64),
        masses,
    )

    # Then: 선형·각운동량이 새로 생성되지 않아야 한다
    initial_momenta = np.concatenate(
        (initial_linear, initial_angular),
    )
    final_momenta = np.concatenate(
        (final_linear, final_angular),
    )

    np.testing.assert_allclose(
        final_momenta,
        initial_momenta,
        atol=1e-5,
        rtol=0.0,
    )
