import numpy as np

from cloth.grid import create_cloth_grid


def test_10x10_grid_should_have_expected_counts_and_valid_indices() -> None:
    """10x10 격자를 생성하면 요소 개수가 정확하고 인덱스가 유효해야 한다."""
    # Given: 한 변에 정점이 10개인 격자 크기
    grid_size = 10

    # When: 천 메쉬를 생성하면
    vertices, triangles, edges = create_cloth_grid(grid_size)

    # Then: 정점, 삼각형 인덱스, 엣지 인덱스 개수가 정확해야 한다
    assert vertices.shape == (100, 3)
    assert triangles.shape == (162 * 3,)
    assert edges.shape == (261 * 2,)

    # Then: 모든 삼각형 인덱스가 존재하는 정점을 가리켜야 한다
    assert triangles.min() == 0
    assert triangles.max() == 99


def test_10x10_grid_should_be_flat_and_centered_at_origin() -> None:
    """10x10 격자를 생성하면 원점 중심의 x-y 평면 위에 놓여야 한다."""
    # Given: 한 변에 정점이 10개인 격자 크기
    grid_size = 10

    # When: 천 메쉬를 생성하면
    vertices, _, _ = create_cloth_grid(grid_size)

    # Then: 모든 정점의 z 좌표가 0이어야 한다
    np.testing.assert_allclose(vertices[:, 2], 0.0)

    # Then: x와 y 좌표 범위가 각각 -1부터 1까지여야 한다
    np.testing.assert_allclose(vertices[:, 0].min(), -1.0)
    np.testing.assert_allclose(vertices[:, 0].max(), 1.0)
    np.testing.assert_allclose(vertices[:, 1].min(), -1.0)
    np.testing.assert_allclose(vertices[:, 1].max(), 1.0)

    # Then: 첫 번째 행은 위쪽에 있어야 한다
    np.testing.assert_allclose(vertices[:grid_size, 1], 1.0)
