"""천의 물리 상태와 시간 적분을 관리한다."""

import numpy as np
import taichi as ti

from cloth.constraint import IdentityConstraintSet, ProjectiveConstraintSet
from cloth.global_system import DiagonalGlobalSystem
from cloth.rest_state import TriangleRestState
from lib.taichi_typing import TaichiF32, TaichiVector3F32


@ti.data_oriented
class ClothSolver:
    def __init__(
        self,
        initial_positions: np.ndarray,
        rest_state: TriangleRestState,
        time_step: float,
    ) -> None:
        self._initial_positions = np.asarray(
            initial_positions,
            dtype=np.float32,
        ).copy()

        vertex_count = len(self._initial_positions)
        triangle_count = len(rest_state.triangles)

        # q_n, v, s, q_n+1(구하는 중)
        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self.predicted_positions = ti.Vector.field(
            3,
            dtype=ti.f32,
            shape=vertex_count,
        )
        self.solve_positions = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)

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

        self._solver_iterations = 1  # local <-> global 반복 횟수

        # Constraints
        self.identity_constraints = IdentityConstraintSet(vertex_count, weight=1.0)
        self.projective_constraints: list[ProjectiveConstraintSet] = [
            self.identity_constraints  # 더미 ConstraintSet
        ]

        # Global System
        self.global_system = DiagonalGlobalSystem(
            vertex_count=vertex_count,
            time_step=time_step,
            constraints=self.projective_constraints,
        )
        self._time_step = time_step

    def reset(self) -> None:
        """위치와 속도를 시뮬레이션 시작 상태로 되돌린다."""
        self.positions.from_numpy(self._initial_positions)
        self.velocities.fill(0.0)
        self.predicted_positions.from_numpy(self._initial_positions)
        self.solve_positions.from_numpy(self._initial_positions)

    def step(
        self,
        gravity: tuple[float, float, float],
    ) -> None:
        """주어진 시간만큼 물리 상태를 진행한다."""
        self._predict_positions(self._time_step, gravity)
        self._initialize_solve_positions()
        for _ in range(self._solver_iterations):
            self._local_step()
            self._global_step()

            # 'collision'을 여기서 따로? self._project_collisions()
            self._project_ground_constraint()
        self._update_state(self._time_step)

    @ti.kernel
    def _predict_positions(
        self,
        time_step: TaichiF32,
        gravity: TaichiVector3F32,
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
    def _initialize_solve_positions(self):
        for vertex_index in self.solve_positions:
            self.solve_positions[vertex_index] = self.predicted_positions[vertex_index]

    def _local_step(self) -> None:
        # Constraint Projection..
        # 각 삼각형에 대해, p_i를 찾기
        # self._projected_deformations, self._projected_bending, ...
        for constraint in self.projective_constraints:
            constraint.project(self.solve_positions)

    def _global_step(self) -> None:
        self.global_system.solve(
            predicted_positions=self.predicted_positions,
            solve_positions=self.solve_positions,
        )

    @ti.kernel
    def _project_ground_constraint(self):
        # 모든 정점이 지면 아래로 내려가지 않도록 제한한다.
        for vertex_index in self.solve_positions:
            solved = self.solve_positions[vertex_index]
            if solved[1] < 0.0:
                solved[1] = 0.0
                self.solve_positions[vertex_index] = solved
                # solved는 참조가 아니라서 이렇게 업데이트해줘야

    @ti.kernel
    def _update_state(self, time_step: TaichiF32):
        # 모든 정점의 위치와 속도를 업데이트한다.
        for vertex_index in self.positions:
            new_velocity = (
                self.solve_positions[vertex_index] - self.positions[vertex_index]
            ) / time_step

            self.velocities[vertex_index] = new_velocity
            self.positions[vertex_index] = self.solve_positions[vertex_index]
