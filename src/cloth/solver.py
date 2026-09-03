"""천의 물리 상태와 시간 적분을 관리한다."""

import numpy as np
import taichi as ti

from cloth.constraint import (
    PositionConstraintSet,
    ProjectiveConstraintSet,
    StrainConstraintSet,
)
from cloth.global_system import DenseGlobalSystem
from cloth.rest_state import TriangleRestState
from lib.taichi_typing import TaichiF32, TaichiVector3F32

POSITIONS_CONSTRAINT_WEIGHT = 1_000_000.0


@ti.data_oriented
class ClothSolver:
    def __init__(
        self,
        initial_positions: np.ndarray,
        rest_state: TriangleRestState,
        time_step: float,
        fixed_vertex_indices: np.ndarray | None = None,
    ) -> None:
        self._initial_positions = np.asarray(
            initial_positions,
            dtype=np.float32,
        ).copy()
        self._current_positions_numpy = self._initial_positions.copy()

        vertex_count = len(self._initial_positions)
        self._vertex_count = vertex_count
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
        self._vertex_accelerations = ti.Vector.field(
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

        self._solver_iterations = 5  # local <-> global 반복 횟수

        # Constraints
        self.strain_constraints = StrainConstraintSet(
            rest_state=rest_state,
            weight=10_000.0,
            minimum_strain=0.9,
            maximum_strain=1.1,
        )
        self.projective_constraints: list[ProjectiveConstraintSet] = [
            self.strain_constraints  # Strain ConstraintSet
        ]

        if fixed_vertex_indices is not None:
            self.positions_constraints = PositionConstraintSet(
                vertex_indices=fixed_vertex_indices,
                target_positions=self._initial_positions[fixed_vertex_indices],
                weight=POSITIONS_CONSTRAINT_WEIGHT,
            )
            self.projective_constraints.append(self.positions_constraints)

        # Global System
        self.global_system = DenseGlobalSystem(
            vertex_count=vertex_count,
            time_step=time_step,
            constraints=self.projective_constraints,
        )
        self._time_step = time_step

    def reset(self) -> None:
        """위치와 속도를 시뮬레이션 시작 상태로 되돌린다."""
        self._current_positions_numpy = self._initial_positions.copy()
        self.positions.from_numpy(self._initial_positions)
        self.velocities.fill(0.0)
        self.predicted_positions.from_numpy(self._initial_positions)
        self.solve_positions.from_numpy(self._initial_positions)
        self._vertex_accelerations.fill(0.0)

    @property
    def current_positions(self) -> np.ndarray:
        """CPU 연산용 현재 위치의 읽기 전용 view를 반환한다."""
        positions = self._current_positions_numpy.view()
        positions.flags.writeable = False
        return positions

    def step(
        self,
        gravity: tuple[float, float, float],
        vertex_accelerations: np.ndarray | None = None,
    ) -> None:
        """주어진 시간만큼 물리 상태를 진행한다."""
        self._set_vertex_accelerations(vertex_accelerations)
        self._predict_positions(self._time_step, gravity)

        # GPU -> CPU 복사는 physics step당 한 번만 한다. 이후 Local-Global
        # 반복은 같은 float64 NumPy 배열을 계속 사용한다.
        predicted_positions = self.predicted_positions.to_numpy().astype(np.float64)
        solve_positions = predicted_positions.copy()
        for _ in range(self._solver_iterations):
            self._local_step(solve_positions)
            solve_positions = self._global_step(predicted_positions)

            # 'collision'을 여기서 따로? self._project_collisions()
            self._project_ground_constraint(solve_positions)

        # 반복 도중에는 업로드하지 않고 최종 해만 Taichi 상태에 반영한다.
        final_positions = solve_positions.astype(np.float32)
        self._current_positions_numpy = final_positions
        self.solve_positions.from_numpy(final_positions)
        self._update_state(self._time_step)

    def _set_vertex_accelerations(
        self,
        vertex_accelerations: np.ndarray | None,
    ) -> None:
        if vertex_accelerations is None:
            self._vertex_accelerations.fill(0.0)
            return

        accelerations_numpy = np.asarray(vertex_accelerations, dtype=np.float32)
        expected_shape = (self._vertex_count, 3)
        if accelerations_numpy.shape != expected_shape:
            raise ValueError(f"vertex_accelerations must have shape {expected_shape}")
        if not np.all(np.isfinite(accelerations_numpy)):
            raise ValueError("vertex_accelerations must contain only finite values")

        self._vertex_accelerations.from_numpy(accelerations_numpy)

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
            acceleration = gravity + self._vertex_accelerations[vertex_index]
            predicted = (
                position + velocity * time_step + time_step * time_step * acceleration
            )
            self.predicted_positions[vertex_index] = predicted

    def _local_step(self, solve_positions: np.ndarray) -> None:
        # Constraint Projection..
        # 각 삼각형에 대해, p_i를 찾기
        # self._projected_deformations, self._projected_bending, ...
        for constraint in self.projective_constraints:
            constraint.project(solve_positions)

    def _global_step(self, predicted_positions: np.ndarray) -> np.ndarray:
        return self.global_system.solve(predicted_positions=predicted_positions)

    @staticmethod
    def _project_ground_constraint(solve_positions: np.ndarray) -> None:
        # 모든 정점이 지면 아래로 내려가지 않도록 제한한다.
        solve_positions[:, 1] = np.maximum(solve_positions[:, 1], 0.0)

    @ti.kernel
    def _update_state(self, time_step: TaichiF32):
        # 모든 정점의 위치와 속도를 업데이트한다.
        for vertex_index in self.positions:
            new_velocity = (
                self.solve_positions[vertex_index] - self.positions[vertex_index]
            ) / time_step

            self.velocities[vertex_index] = new_velocity
            self.positions[vertex_index] = self.solve_positions[vertex_index]
