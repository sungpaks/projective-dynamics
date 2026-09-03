import numpy as np

from interaction.screen_space_acceleration import (
    calculate_screen_space_accelerations,
)


def test_acceleration_should_fall_off_from_cursor_center() -> None:
    """가속도는 클릭 중심에서 최대이고 반지름 경계에서 0이어야 한다."""
    # Given: 중심, 반지름 중간, 경계, 외부에 투영되는 네 정점
    positions = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.75, 0.0, 0.0),
        ),
        dtype=np.float64,
    )

    # When: 중심 가속도 10, 반지름 50 pixel인 가속도장을 계산하면
    accelerations = calculate_screen_space_accelerations(
        positions=positions,
        cursor_position=(0.5, 0.5),
        view_matrix=np.eye(4),
        projection_matrix=np.eye(4),
        window_size=(200, 100),
        radius_pixels=50.0,
        magnitude=10.0,
        direction=np.array((0.0, 0.0, 2.0)),
    )

    # Then: 방향은 정규화되고 거리별 compact falloff가 적용되어야 한다
    expected = np.array(
        (
            (0.0, 0.0, 10.0),
            (0.0, 0.0, 5.625),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    np.testing.assert_allclose(accelerations, expected, atol=1e-6)
    assert accelerations.dtype == np.float32


def test_acceleration_radius_should_be_circular_in_pixel_space() -> None:
    """창의 종횡비와 무관하게 같은 pixel 거리는 같은 가속도를 받아야 한다."""
    # Given: 중심에서 각각 가로와 세로로 25 pixel 떨어져 투영되는 정점
    positions = np.array(
        (
            (0.25, 0.0, 0.0),
            (0.0, 0.5, 0.0),
        ),
        dtype=np.float64,
    )

    # When: 가로세로 크기가 다른 창에서 원형 가속도장을 계산하면
    accelerations = calculate_screen_space_accelerations(
        positions=positions,
        cursor_position=(0.5, 0.5),
        view_matrix=np.eye(4),
        projection_matrix=np.eye(4),
        window_size=(200, 100),
        radius_pixels=50.0,
        magnitude=10.0,
        direction=np.array((1.0, 0.0, 0.0)),
    )

    # Then: 두 정점은 같은 크기의 가속도를 받아야 한다
    np.testing.assert_allclose(accelerations[0], accelerations[1], atol=1e-6)


def test_vertices_behind_camera_should_receive_no_acceleration() -> None:
    """카메라 뒤쪽 정점은 화면상 중심과 겹쳐도 영향을 받지 않아야 한다."""
    # Given: Taichi의 row-vector convention에서 카메라 앞뒤에 놓인 정점
    positions = np.array(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 10.0),
        ),
        dtype=np.float64,
    )
    view_matrix = np.eye(4)
    view_matrix[3, 2] = -5.0
    projection_matrix = np.zeros((4, 4), dtype=np.float64)
    projection_matrix[0, 0] = 1.0
    projection_matrix[1, 1] = 1.0
    projection_matrix[2, 2] = 0.0001
    projection_matrix[2, 3] = -1.0
    projection_matrix[3, 2] = 0.1

    # When: 두 정점에 대해 화면 공간 가속도장을 계산하면
    accelerations = calculate_screen_space_accelerations(
        positions=positions,
        cursor_position=(0.5, 0.5),
        view_matrix=view_matrix,
        projection_matrix=projection_matrix,
        window_size=(200, 100),
        radius_pixels=50.0,
        magnitude=10.0,
        direction=np.array((0.0, 0.0, 1.0)),
    )

    # Then: 앞쪽 정점만 가속도를 받아야 한다
    np.testing.assert_allclose(accelerations[0], (0.0, 0.0, 10.0), atol=1e-6)
    np.testing.assert_array_equal(accelerations[1], (0.0, 0.0, 0.0))


def test_acceleration_field_should_reject_invalid_inputs() -> None:
    """계산 의미를 정의할 수 없는 입력은 명확하게 거부해야 한다."""
    # Given: 한 정점과 기본 카메라 행렬
    positions = np.zeros((1, 3), dtype=np.float64)
    common_arguments = {
        "positions": positions,
        "cursor_position": (0.5, 0.5),
        "view_matrix": np.eye(4),
        "projection_matrix": np.eye(4),
        "window_size": (200, 100),
        "magnitude": 10.0,
    }

    # When/Then: 반지름이 0이면 ValueError가 발생해야 한다
    with np.testing.assert_raises_regex(ValueError, "radius_pixels must be positive"):
        calculate_screen_space_accelerations(
            **common_arguments,
            radius_pixels=0.0,
            direction=np.array((0.0, 0.0, 1.0)),
        )

    # When/Then: 방향이 영벡터이면 ValueError가 발생해야 한다
    with np.testing.assert_raises_regex(ValueError, "direction must be a non-zero"):
        calculate_screen_space_accelerations(
            **common_arguments,
            radius_pixels=50.0,
            direction=np.zeros(3),
        )

    # When/Then: 정점 shape가 잘못되면 ValueError가 발생해야 한다
    with np.testing.assert_raises_regex(ValueError, "positions must have shape"):
        calculate_screen_space_accelerations(
            **(common_arguments | {"positions": np.zeros((3,))}),
            radius_pixels=50.0,
            direction=np.array((0.0, 0.0, 1.0)),
        )
