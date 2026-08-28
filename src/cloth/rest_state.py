from dataclasses import dataclass

import numpy as np

epsilon = 1e-8  # 삼각형 면적이 이보다 작으면 degenerate로 간주


# tuple로 여러 배열 반환할 수도 있지만 알기 쉽게 dataclass
@dataclass(frozen=True)
class TriangleRestState:
    triangles: np.ndarray  # shape: (triangle_count, 3)
    edge_matrices: np.ndarray  # shape: (triangle_count, 2, 2)
    inverse_edge_matrices: np.ndarray  # shape: (triangle_count, 2, 2)
    areas: np.ndarray  # shape: (triangle_count,)


def create_triangle_rest_state(
    rest_positions: np.ndarray, triangle_indices: np.ndarray
) -> TriangleRestState:
    rest_positions = np.asarray(rest_positions, dtype=np.float32)

    if rest_positions.ndim != 2 or rest_positions.shape[1] != 3:
        raise ValueError(
            f"rest_positions must be a 2D array with shape (N, 3), \
              got shape {rest_positions.shape}"
        )

    triangles = np.asarray(triangle_indices, dtype=np.int32).reshape(
        -1, 3
    )  # 3개씩 삼각형으로 묶음. `-1`: 자동으로 행(삼각형) 개수 계산

    triangle_count = len(triangles)

    # empty: 초기화하지 않은 배열. 원소는 이후 loop에서 채워진다
    edge_matrices = np.empty((triangle_count, 2, 2), dtype=np.float32)

    inverse_edge_matrices = np.empty((triangle_count, 2, 2), dtype=np.float32)

    areas = np.empty(triangle_count, dtype=np.float32)

    for triangle_index, triangle in enumerate(triangles):
        i, j, k = triangle

        q_i = rest_positions[i]
        q_j = rest_positions[j]
        q_k = rest_positions[k]

        edge_1 = q_j - q_i
        edge_2 = q_k - q_i

        edge_1_length = np.linalg.norm(edge_1)
        cross_product = np.cross(edge_1, edge_2)
        twice_area = np.linalg.norm(cross_product)

        if edge_1_length <= epsilon or twice_area <= epsilon:
            raise ValueError(
                f"Degenerate triangle at index {triangle_index}: "
                f"edge length = {edge_1_length}, area = {0.5 * twice_area}"
            )

        # 첫 edge를 로컬 x축
        local_x = edge_1 / edge_1_length
        # 삼각형의 법선은 cross로
        local_normal = cross_product / twice_area
        # 로컬 y축은 노멀과 로컬 x축의 외적
        local_y = np.cross(local_normal, local_x)

        # 삼각형 크기 = 외적 크기의 절반
        area = 0.5 * twice_area

        # edge matrix: 로컬 x-y 좌표계에서의 edge 좌표
        rest_edge_matrix = np.array(
            (
                (
                    np.dot(edge_1, local_x),
                    np.dot(edge_2, local_x),
                ),
                (
                    np.dot(edge_1, local_y),
                    np.dot(edge_2, local_y),
                ),
            ),
            dtype=np.float32,
        )

        edge_matrices[triangle_index] = rest_edge_matrix

        inverse_edge_matrices[triangle_index] = np.linalg.inv(rest_edge_matrix).astype(
            np.float32
        )

        areas[triangle_index] = area

    return TriangleRestState(
        triangles=triangles.copy(),
        edge_matrices=edge_matrices,
        inverse_edge_matrices=inverse_edge_matrices,
        areas=areas,
    )
