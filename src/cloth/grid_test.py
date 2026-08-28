import numpy as np

from cloth.grid import create_cloth_grid


def test_10x10_grid_should_have_expected_counts_and_valid_indices() -> None:
    """10x10 격자를 생성하면 요소 개수가 정확하고 인덱스가 유효해야 한다."""
    # 준비: 한 변에 정점이 10개인 격자 크기
    grid_size = 10

    # 실행: 천 메쉬를 생성하면
    vertices, triangles, edges = create_cloth_grid(grid_size)

    # 검증: 정점, 삼각형 인덱스, 엣지 인덱스 개수가 정확해야 한다
    assert vertices.shape == (100, 3)
    assert triangles.shape == (162 * 3,)
    assert edges.shape == (261 * 2,)

    # 검증: 모든 삼각형 인덱스가 존재하는 정점을 가리켜야 한다
    assert triangles.min() == 0
    assert triangles.max() == 99


def test_10x10_grid_should_follow_position_and_normal() -> None:
    """10x10 격자를 생성하면 전달한 위치와 법선을 따라 배치되어야 한다."""
    # 준비: 한 변에 정점이 10개인 격자 크기
    grid_size = 10
    position = np.array((2.0, -1.0, 3.0), dtype=np.float32)
    normal = np.array((1.0, 0.0, 1.0), dtype=np.float32)
    unit_normal = normal / np.linalg.norm(normal)

    # 실행: 천 메쉬를 생성하면
    vertices, _, _ = create_cloth_grid(
        grid_size,
        position=tuple(position),
        normal=tuple(normal),
    )

    # 검증: 격자의 중심은 전달한 위치여야 한다
    np.testing.assert_allclose(vertices.mean(axis=0), position, atol=1e-6)

    # 검증: 모든 정점은 전달한 법선에 수직인 평면 위에 있어야 한다
    offsets = vertices - position
    np.testing.assert_allclose(offsets @ unit_normal, 0.0, atol=1e-6)

    # 검증: 각 로컬 축의 전체 길이는 2여야 한다
    np.testing.assert_allclose(
        np.linalg.norm(vertices[grid_size - 1] - vertices[0]), 2.0
    )
    np.testing.assert_allclose(np.linalg.norm(vertices[-grid_size] - vertices[0]), 2.0)


def test_grid_should_reject_zero_normal() -> None:
    """방향을 정의할 수 없는 영벡터 법선은 거부해야 한다."""
    with np.testing.assert_raises_regex(ValueError, "normal must be a non-zero vector"):
        create_cloth_grid(10, normal=(0.0, 0.0, 0.0))
