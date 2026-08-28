import numpy as np

from cloth.grid import create_cloth_grid
from cloth.rest_state import create_triangle_rest_state


def test_10x10_rest_state_should_have_expected_shapes() -> None:
    """10x10 격자의 삼각형별 rest data는 예상 shape를 가져야 한다."""
    # Given: 한 변에 정점이 10개인 천 격자
    vertices, triangle_indices, _ = create_cloth_grid(10)

    # When: 격자의 초기 위치와 삼각형으로 rest state를 생성하면
    rest_state = create_triangle_rest_state(vertices, triangle_indices)

    # Then: 162개 삼각형의 인덱스, edge matrix, 역행렬, 면적이 저장되어야 한다
    assert rest_state.triangles.shape == (162, 3)
    assert rest_state.edge_matrices.shape == (162, 2, 2)
    assert rest_state.inverse_edge_matrices.shape == (162, 2, 2)
    assert rest_state.areas.shape == (162,)


def test_rest_state_should_preserve_total_grid_area() -> None:
    """Rest state의 삼각형 면적 합은 초기 격자 면적과 같아야 한다."""
    # Given: 가로와 세로 길이가 각각 2인 천 격자
    vertices, triangle_indices, _ = create_cloth_grid(10)

    # When: 격자의 triangle rest state를 생성하면
    rest_state = create_triangle_rest_state(vertices, triangle_indices)

    # Then: 모든 rest triangle의 면적 합은 4여야 한다
    np.testing.assert_allclose(rest_state.areas.sum(), 4.0, atol=1e-5)


def test_rest_state_should_reject_degenerate_triangle() -> None:
    """면적이 0인 삼각형은 rest mapping을 정의할 수 없어야 한다."""
    # Given: 한 직선 위에 있어 면적이 0인 세 정점
    rest_positions = np.array(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    triangle_indices = np.array((0, 1, 2), dtype=np.int32)

    # When/Then: rest state를 생성하면 degenerate 오류가 발생해야 한다
    with np.testing.assert_raises_regex(
        ValueError,
        "Degenerate triangle at index 0",
    ):
        create_triangle_rest_state(rest_positions, triangle_indices)
