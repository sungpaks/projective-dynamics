"""천 시뮬레이션용 절차적 격자 메쉬 생성."""

import numpy as np


def create_cloth_basis(
    normal: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """법선으로부터 천의 로컬 X, Y, Z축을 계산한다."""
    local_z = np.asarray(normal, dtype=np.float32)
    normal_length = np.linalg.norm(local_z)
    if normal_length == 0.0:
        raise ValueError("normal must be a non-zero vector")
    local_z /= normal_length

    # 로컬 Y축을 가능한 한 월드 Y축과 일치시킨다. 법선이 월드 Y축과 평행하면
    # 외적이 영벡터가 되는 것을 피하기 위해 월드 Z축을 기준으로 사용한다.
    reference_up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    if abs(float(np.dot(reference_up, local_z))) > 0.999:
        reference_up = np.array((0.0, 0.0, 1.0), dtype=np.float32)

    local_x = np.cross(reference_up, local_z)
    local_x /= np.linalg.norm(local_x)
    local_y = np.cross(local_z, local_x)
    return local_x, local_y, local_z


def create_cloth_grid(
    size: int,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``position``을 중심으로 ``normal`` 방향을 바라보는 격자를 생성한다."""
    vertex_count = size * size
    cell_count = (size - 1) * (size - 1)

    position_vector = np.asarray(position, dtype=np.float32)
    local_x, local_y, _ = create_cloth_basis(normal)

    # 각 정점은 32비트 실수로 된 (x, y, z) 위치를 저장한다.
    vertices = np.zeros((vertex_count, 3), dtype=np.float32)

    for row in range(size):
        for column in range(size):
            vertex_index = row * size + column

            # 첫 번째 행이 위쪽에 오도록 로컬 x-y 공간에서 생성한 뒤,
            # 해당 로컬 좌표계를 월드 공간에 배치한다.
            x = column / (size - 1) * 2.0 - 1.0
            y = 1.0 - row / (size - 1) * 2.0
            vertices[vertex_index] = position_vector + x * local_x + y * local_y

    # 정사각형 셀 하나를 삼각형 두 개로 나누며, 삼각형마다 인덱스 세 개를 사용한다.
    indices = np.zeros(cell_count * 2 * 3, dtype=np.int32)

    write_index = 0
    for row in range(size - 1):
        for column in range(size - 1):
            top_left = row * size + column
            top_right = top_left + 1
            bottom_left = (row + 1) * size + column
            bottom_right = bottom_left + 1

            indices[write_index : write_index + 6] = (
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            )
            write_index += 6

    # scene.lines()에는 정점 인덱스 쌍이 필요하며, 집합으로 공유 엣지를 제거한다.
    edges: set[tuple[int, int]] = set()
    for triangle in indices.reshape(-1, 3):
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            start_index = int(start)
            end_index = int(end)
            edge = (
                min(start_index, end_index),
                max(start_index, end_index),
            )
            edges.add(edge)

    edge_indices = np.array(sorted(edges), dtype=np.int32).reshape(-1)

    return vertices, indices, edge_indices
