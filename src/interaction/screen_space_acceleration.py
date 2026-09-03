"""화면상의 원형 영역으로부터 정점별 가속도장을 계산한다."""

import numpy as np

_MINIMUM_CLIP_W = 1e-12


def calculate_screen_space_accelerations(
    positions: np.ndarray,
    cursor_position: tuple[float, float],
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    window_size: tuple[int, int],
    radius_pixels: float,
    magnitude: float,
    direction: np.ndarray,
) -> np.ndarray:
    """
    화면 상에서 클릭 위치를 중심으로 원형 영역에 가속도 field를 생성하고,
    이에 영향을 받는 정점들이 받는 가속도를 계산한다.
    원형 가속도 field는 radius_pixels 내에서 중심에 가까울 수록 강하다

    반환 shape는 (vertex_count, 3).

    Ray casting이나 depth test를 수행하지 않으므로
    화면에서 겹쳐 보이는 여러 겹의 표면이 함께 영향을 받을 수 있다.
    """
    positions_numpy = _validate_positions(positions)
    cursor_numpy = _validate_cursor_position(cursor_position)
    view_numpy = _validate_matrix(view_matrix, "view_matrix")
    projection_numpy = _validate_matrix(projection_matrix, "projection_matrix")
    window_size_numpy = _validate_window_size(window_size)
    radius = _validate_radius(radius_pixels)
    acceleration_magnitude = _validate_magnitude(magnitude)
    unit_direction = _normalize_direction(direction)

    homogeneous_positions = _to_homogeneous_positions(positions_numpy)
    clip_positions = _project_to_clip_space(
        homogeneous_positions,
        view_numpy,
        projection_numpy,
    )
    screen_pixels, visible = _clip_to_screen_pixels(
        clip_positions,
        window_size_numpy,
    )
    cursor_pixels = _cursor_to_pixels(cursor_numpy, window_size_numpy)
    falloff = _calculate_radial_falloff(
        screen_pixels,
        cursor_pixels,
        radius,
    )
    falloff[~visible] = 0.0

    accelerations = falloff[:, None] * acceleration_magnitude * unit_direction[None, :]
    return accelerations.astype(np.float32)


def _validate_positions(positions: np.ndarray) -> np.ndarray:
    positions_numpy = np.asarray(positions, dtype=np.float64)
    if positions_numpy.ndim != 2 or positions_numpy.shape[1] != 3:
        raise ValueError("positions must have shape (vertex_count, 3)")
    if not np.all(np.isfinite(positions_numpy)):
        raise ValueError("positions must contain only finite values")
    return positions_numpy


def _validate_cursor_position(
    cursor_position: tuple[float, float],
) -> np.ndarray:
    cursor_numpy = np.asarray(cursor_position, dtype=np.float64)
    if cursor_numpy.shape != (2,):
        raise ValueError("cursor_position must have shape (2,)")
    if not np.all(np.isfinite(cursor_numpy)):
        raise ValueError("cursor_position must contain only finite values")
    if np.any(cursor_numpy < 0.0) or np.any(cursor_numpy > 1.0):
        raise ValueError("cursor_position must be in the range [0, 1]")
    return cursor_numpy


def _validate_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    matrix_numpy = np.asarray(matrix, dtype=np.float64)
    if matrix_numpy.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(matrix_numpy)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix_numpy


def _validate_window_size(window_size: tuple[int, int]) -> np.ndarray:
    window_size_numpy = np.asarray(window_size, dtype=np.float64)
    if window_size_numpy.shape != (2,):
        raise ValueError("window_size must have shape (2,)")
    if not np.all(np.isfinite(window_size_numpy)) or np.any(window_size_numpy <= 0.0):
        raise ValueError("window_size must contain positive values")
    return window_size_numpy


def _validate_radius(radius_pixels: float) -> float:
    radius = float(radius_pixels)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius_pixels must be positive and finite")
    return radius


def _validate_magnitude(magnitude: float) -> float:
    acceleration_magnitude = float(magnitude)
    if not np.isfinite(acceleration_magnitude) or acceleration_magnitude < 0.0:
        raise ValueError("magnitude must be non-negative and finite")
    return acceleration_magnitude


def _normalize_direction(direction: np.ndarray) -> np.ndarray:
    direction_numpy = np.asarray(direction, dtype=np.float64)
    if direction_numpy.shape != (3,):
        raise ValueError("direction must have shape (3,)")
    if not np.all(np.isfinite(direction_numpy)):
        raise ValueError("direction must contain only finite values")

    direction_length = np.linalg.norm(direction_numpy)
    if direction_length == 0.0:
        raise ValueError("direction must be a non-zero vector")
    return direction_numpy / direction_length


def _to_homogeneous_positions(positions: np.ndarray) -> np.ndarray:
    homogeneous_component = np.ones((len(positions), 1), dtype=np.float64)
    return np.concatenate((positions, homogeneous_component), axis=1)


def _project_to_clip_space(
    homogeneous_positions: np.ndarray,
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
) -> np.ndarray:
    # Taichi Camera가 반환하는 행렬은 row-vector convention을 사용한다.
    return homogeneous_positions @ view_matrix @ projection_matrix


def _clip_to_screen_pixels(
    clip_positions: np.ndarray,
    window_size: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    visible = clip_positions[:, 3] > _MINIMUM_CLIP_W
    ndc_positions = np.zeros((len(clip_positions), 2), dtype=np.float64)
    np.divide(
        clip_positions[:, :2],
        clip_positions[:, 3, None],
        out=ndc_positions,
        where=visible[:, None],
    )
    normalized_screen_positions = ndc_positions * 0.5 + 0.5
    return normalized_screen_positions * window_size[None, :], visible


def _cursor_to_pixels(
    cursor_position: np.ndarray,
    window_size: np.ndarray,
) -> np.ndarray:
    return cursor_position * window_size


def _calculate_radial_falloff(
    screen_pixels: np.ndarray,
    cursor_pixels: np.ndarray,
    radius_pixels: float,
) -> np.ndarray:
    offsets = screen_pixels - cursor_pixels[None, :]
    squared_distances = np.sum(offsets * offsets, axis=1)
    squared_radius = radius_pixels * radius_pixels

    falloff = np.zeros(len(screen_pixels), dtype=np.float64)
    inside = squared_distances < squared_radius
    normalized_squared_distances = squared_distances[inside] / squared_radius
    falloff[inside] = (1.0 - normalized_squared_distances) ** 2
    return falloff
