import numpy as np
import taichi as ti

from cloth.grid import create_cloth_grid
from cloth.rest_state import create_triangle_rest_state
from cloth.solver import ClothSolver


def test_uniform_vertex_acceleration_should_translate_rest_cloth() -> None:
    """균일한 정점별 가속도는 rest cloth 전체를 같은 만큼 이동시켜야 한다."""
    # Given: 변형과 초기 속도가 없는 천과 균일한 z축 가속도
    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False)
    initial_positions, triangle_indices, _ = create_cloth_grid(
        3,
        position=(0.0, 2.0, 0.0),
    )
    rest_state = create_triangle_rest_state(initial_positions, triangle_indices)
    time_step = 1.0 / 60.0
    solver = ClothSolver(initial_positions, rest_state, time_step=time_step)
    vertex_accelerations = np.full_like(initial_positions, (0.0, 0.0, 12.0))

    # When: 외력 없이 정점별 가속도만 적용하여 한 step 진행하면
    solver.step(
        gravity=(0.0, 0.0, 0.0),
        vertex_accelerations=vertex_accelerations,
    )

    # Then: 모든 정점은 h²a만큼 평행 이동해야 한다
    expected_positions = initial_positions + time_step**2 * vertex_accelerations
    np.testing.assert_allclose(
        solver.positions.to_numpy(),
        expected_positions,
        atol=1e-5,
        rtol=0.0,
    )


def test_solver_should_reject_invalid_vertex_acceleration_shape() -> None:
    """정점별 가속도 shape가 mesh와 다르면 명확하게 거부해야 한다."""
    # Given: 3x3 천과 정점 수가 맞지 않는 가속도 배열
    ti.reset()
    ti.init(arch=ti.cpu, offline_cache=False)
    initial_positions, triangle_indices, _ = create_cloth_grid(3)
    rest_state = create_triangle_rest_state(initial_positions, triangle_indices)
    solver = ClothSolver(initial_positions, rest_state, time_step=1.0 / 60.0)
    invalid_accelerations = np.zeros((8, 3), dtype=np.float32)

    # When/Then: 한 step에 잘못된 가속도를 주입하면 ValueError가 발생해야 한다
    with np.testing.assert_raises_regex(
        ValueError,
        "vertex_accelerations must have shape",
    ):
        solver.step(
            gravity=(0.0, 0.0, 0.0),
            vertex_accelerations=invalid_accelerations,
        )
