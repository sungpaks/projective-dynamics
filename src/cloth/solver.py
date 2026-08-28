# pyright: reportInvalidTypeForm=false

"""천의 물리 상태와 시간 적분을 관리한다."""

import numpy as np
import taichi as ti

from cloth.rest_state import TriangleRestState


@ti.data_oriented
class ClothSolver:
    def __init__(self, initial_positions: np.ndarray, rest_state: TriangleRestState):
        self._initial_positions = np.asarray(
            initial_positions,
            dtype=np.float32,
        ).copy()

        vertex_count = len(self._initial_positions)
        triangle_count = len(rest_state.triangles)

        # q, v, s
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.predicted_positions = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=vertex_count,
        )

        self.reset()

        # 삼각형 인덱스: 정점 인덱스 세 개씩
        self.triangles = ti.Vector.field(
            3,
            dtype=ti.i32,
            shape=triangle_count,
        )
        # X_g의 역행렬: 삼각형 하나의 Edge 두 개를 갖는 2x2 행렬
        self.inverse_rest_matrices = ti.Matrix.field(
            2,
            2,
            dtype=ti.f32,
            shape=triangle_count,
        )
        # 삼각형 면적: 삼각형 하나당 하나의 scalar(float)
        self.areas = ti.field(dtype=ti.f32, shape=triangle_count)

        self.triangles.from_numpy(rest_state.triangles)
        self.inverse_rest_matrices.from_numpy(rest_state.inverse_edge_matrices)
        self.areas.from_numpy(rest_state.areas)

    def reset(self) -> None:
        """위치와 속도를 시뮬레이션 시작 상태로 되돌린다."""
        self.positions.from_numpy(self._initial_positions)
        self.velocities.fill(0.0)
        self.predicted_positions.from_numpy(self._initial_positions)

    def step(
        self,
        time_step: float,
        gravity: tuple[float, float, float],
    ) -> None:
        """주어진 시간만큼 물리 상태를 진행한다."""
        self._predict_positions(time_step, gravity)
        self._project_ground_constraint()
        self._update_state(time_step)

    @ti.kernel
    def _predict_positions(
        self,
        time_step: ti.f32,
        gravity: ti.types.vector(3, ti.f32),
    ):
        # 모든 정점의 예상 위치를 계산한다.
        for vertex_index in self.positions:
            position = self.positions[vertex_index]
            velocity = self.velocities[vertex_index]
            predicted = (
                position + velocity * time_step + time_step * time_step * gravity
            )
            self.predicted_positions[vertex_index] = predicted

    @ti.kernel
    def _project_ground_constraint(self):
        # 모든 정점이 지면 아래로 내려가지 않도록 제한한다.
        for vertex_index in self.predicted_positions:
            predicted = self.predicted_positions[vertex_index]
            if predicted[1] < 0.0:
                predicted[1] = 0.0
                self.predicted_positions[vertex_index] = predicted
                # predicted는 참조가 아니라서 이렇게 업데이트해줘야

    @ti.kernel
    def _update_state(self, time_step: ti.f32):
        # 모든 정점의 위치와 속도를 업데이트한다.
        for vertex_index in self.positions:
            new_velocity = (
                self.predicted_positions[vertex_index] - self.positions[vertex_index]
            ) / time_step

            self.velocities[vertex_index] = new_velocity
            self.positions[vertex_index] = self.predicted_positions[vertex_index]
