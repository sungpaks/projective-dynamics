# pyright: reportInvalidTypeForm=false

"""천의 물리 상태와 시간 적분을 관리한다."""

import numpy as np
import taichi as ti


@ti.data_oriented
class ClothSolver:
    def __init__(self, initial_positions: np.ndarray):
        self._initial_positions = np.asarray(
            initial_positions,
            dtype=np.float32,
        ).copy()
        vertex_count = len(self._initial_positions)

        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.predicted_positions = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=vertex_count,
        )

        self.reset()

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
    def _update_state(self, time_step: ti.f32):
        # 모든 정점의 위치와 속도를 업데이트한다.
        for vertex_index in self.positions:
            new_velocity = (
                self.predicted_positions[vertex_index] - self.positions[vertex_index]
            ) / time_step

            self.velocities[vertex_index] = new_velocity
            self.positions[vertex_index] = self.predicted_positions[vertex_index]
