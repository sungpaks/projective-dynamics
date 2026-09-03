from typing import Protocol

import numpy as np
import taichi as ti

from cloth.matrix_assembler import MatrixAssembler
from cloth.rest_state import TriangleRestState
from lib.taichi_typing import TaichiTemplate


class ProjectiveConstraintSet(Protocol):
    @property
    def instance_count(self) -> int: ...

    def project(self, solve_positions) -> None:
        """Local Step에서 p_i 갱신 (per constraint-instance)"""

    def add_lhs(self, assembler) -> None:
        """고정된 Global Matrix contribution 추가"""

    def add_rhs(self, system_rhs) -> None:
        """현재 p_i로 global RHS contribution 추가"""


@ti.data_oriented
class IdentityConstraintSet:
    """
    더미 ConstraintSet.
    - selection은 각 정점별로
    - 대각 성분 only
    """

    def __init__(self, vertex_count: int, weight: float) -> None:
        self._instance_count = vertex_count
        self.vertex_indices = ti.field(dtype=ti.i32, shape=vertex_count)
        self.projections = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        self._weight = weight

        vertex_indices = np.arange(vertex_count, dtype=np.int32)
        self.vertex_indices.from_numpy(vertex_indices)

    @property
    def instance_count(self) -> int:
        return self._instance_count

    def project(self, solve_positions) -> None:
        self._project_all(solve_positions)

    def add_lhs(self, assembler: MatrixAssembler) -> None:
        # 정점 하나씩 선택되므로 global L matrix의 대각성분 하나에만 영향(weight)을 준다
        for vertex_index in range(self._instance_count):
            assembler.add(vertex_index, vertex_index, self._weight)

    def add_rhs(self, system_rhs) -> None:
        # 정점 하나씩 선택되므로 global rhs의 정점별 항 하나에만 영향(weight)을 준다.
        self._add_rhs(system_rhs)

    @ti.kernel
    def _project_all(self, solve_positions: TaichiTemplate):
        for instance_index in self.vertex_indices:
            # instance_index: constraint instance의 번호
            # vertex_index: instance가 선택한 정점 번호
            # IdentityConstraint는 정점 하나만 선택하므로 instance_index == vertex_index
            vertex_index = self.vertex_indices[instance_index]
            self.projections[instance_index] = solve_positions[vertex_index]

    @ti.kernel
    def _add_rhs(self, system_rhs: TaichiTemplate):
        for instance_index in self.vertex_indices:
            vertex_index = self.vertex_indices[instance_index]
            system_rhs[vertex_index] += self._weight * self.projections[instance_index]


@ti.data_oriented
class StrainConstraintSet:
    def __init__(
        self,
        rest_state: TriangleRestState,
        weight: float,
        minimum_strain: float,
        maximum_strain: float,
    ) -> None:
        triangles_count = len(rest_state.triangles)
        self._instance_count = triangles_count
        self._weight = weight
        self._minimum_strain = minimum_strain
        self._maximum_strain = maximum_strain

        # T_i는 3x2
        self.projections = ti.Matrix.field(3, 2, dtype=ti.f32, shape=triangles_count)
        self._projections_numpy = np.zeros(
            (triangles_count, 3, 2),
            dtype=np.float64,
        )

        self._triangles_numpy = rest_state.triangles.copy()
        self.triangles = ti.Vector.field(3, dtype=ti.i32, shape=triangles_count)
        self.triangles.from_numpy(self._triangles_numpy)

        self._areas_numpy = rest_state.areas.astype(np.float64, copy=True)
        self.areas = ti.field(dtype=ti.f32, shape=triangles_count)
        self.areas.from_numpy(self._areas_numpy.astype(np.float32))

        # triangle 세 위치로부터 두 Edge (q_j - q_i, q_k - q_i)를 계산하는 operator D
        # D = [ [-1, -1], [1, 0], [0, 1] ]^T
        edge_difference_operator_transposed = np.array(
            [
                [-1.0, -1.0],
                [+1.0, 0.0],
                [0.0, +1.0],
            ],
            dtype=np.float64,
        )

        inverse_rest_matrices = rest_state.inverse_edge_matrices.astype(
            np.float64,
            copy=False,
        )

        # gradient coefficient = D^T * X_g^{-1} (X_g: rest edge matrix)
        # shape: (triangle_count, 3, 2)
        self._gradient_coefficients_numpy = (
            edge_difference_operator_transposed[None, :, :] @ inverse_rest_matrices
        )

        self.gradient_coefficients = ti.Matrix.field(
            3,
            2,
            dtype=ti.f32,
            shape=triangles_count,
        )
        self.gradient_coefficients.from_numpy(
            self._gradient_coefficients_numpy.astype(np.float32)
        )
        # numpy배열은 add_lhs()에서, taichi field는 add_rhs()에서 사용

    @property
    def instance_count(self) -> int:
        return self._instance_count

    def project(self, solve_positions) -> None:
        solve_positions_numpy = solve_positions.to_numpy().astype(np.float64)

        # 삼각형 세 정점 위치를 한 번에 모은다 (Selection)
        # shape: (triangle_count, 3, 3)
        triangle_positions = solve_positions_numpy[self._triangles_numpy]

        # triangle 전체에 대해 Q_i^T G_i (batch multiplication)
        # shape: (triangle_count, 3, 2)
        deformation_gradients = (
            triangle_positions.transpose(0, 2, 1) @ self._gradient_coefficients_numpy
        )

        self._projections_numpy = self._project_deformation_gradients(
            deformation_gradients
        )

        # Global solve는 float64 NumPy 원본을 사용한다. Taichi field는
        # 디버깅 및 Taichi 구현과의 비교를 위한 float32 사본이다.
        self.projections.from_numpy(self._projections_numpy.astype(np.float32))

    def add_lhs(self, assembler: MatrixAssembler) -> None:
        # L_i = w_i * A_i * G_i * G_i^T (G_i: gradient coefficient, A_i: triangle area)
        for triangle_index in range(self._instance_count):
            triangle = self._triangles_numpy[triangle_index]
            coefficient = self._gradient_coefficients_numpy[triangle_index]
            area = self._areas_numpy[triangle_index]

            # (3,3)
            local_lhs = self._weight * area * (coefficient @ coefficient.T)

            for local_row in range(3):
                global_row = int(triangle[local_row])

                for local_col in range(3):
                    global_col = int(triangle[local_col])

                    assembler.add(
                        global_row, global_col, local_lhs[local_row, local_col]
                    )

    def add_rhs(self, system_rhs) -> None:
        # self._add_rhs(system_rhs) ti.kernel 대신 numpy계산
        self._add_rhs_numpy(system_rhs)

    def _project_deformation_gradients(
        self,
        deformation_gradients: np.ndarray,
    ) -> np.ndarray:
        """모든 deformation gradient의 singular value를 strain 범위로 투영한다."""
        u, singular_values, vt = np.linalg.svd(
            deformation_gradients,
            full_matrices=False,
        )

        # singular value를 clamp하고
        projected_singular_values = np.clip(
            singular_values,
            self._minimum_strain,
            self._maximum_strain,
        )

        # diag 행렬을 만들지 않고 U의 각 열에 대응 singular value를 곱한다.
        return (u * projected_singular_values[:, None, :]) @ vt

    @ti.kernel
    def _add_rhs(self, system_rhs: TaichiTemplate):
        for triangle_index in range(self._instance_count):
            triangle = self.triangles[triangle_index]  # (3,)
            coefficient = self.gradient_coefficients[triangle_index]  # (3,2)
            projection = self.projections[triangle_index]  # (3,2)
            area = self.areas[triangle_index]  # scalar

            # (3,2) @ (2,3) = (3,3)
            local_rhs = (coefficient @ projection.transpose()) * (self._weight * area)

            for local_vertex_index in ti.static(range(3)):
                global_vertex_index = triangle[local_vertex_index]

                for axis in ti.static(range(3)):
                    # 인접한 triangle들이 같은 global vertex에 동시에 더할 수 있음
                    ti.atomic_add(
                        system_rhs[global_vertex_index][axis],
                        local_rhs[local_vertex_index, axis],  # pyright: ignore[reportIndexIssue]
                    )

    def _add_rhs_numpy(self, system_rhs: np.ndarray) -> None:
        local_rhs = (  # (3, 2) @ (2, 3) = (3, 3)
            self._weight
            * self._areas_numpy[:, None, None]
            * (
                self._gradient_coefficients_numpy
                @ self._projections_numpy.transpose(0, 2, 1)
            )
        )
        for local_vertex_index in range(3):
            np.add.at(
                system_rhs,
                self._triangles_numpy[:, local_vertex_index],
                local_rhs[:, local_vertex_index],
            )


class PositionConstraintSet:
    def __init__(
        self,
        vertex_indices: np.ndarray,
        target_positions: np.ndarray,
        weight: float,
    ) -> None:
        self._vertex_indices = np.asarray(vertex_indices, dtype=np.int32).copy()
        self._target_positions = np.asarray(target_positions, dtype=np.float32).copy()
        self._weight = weight

    @property
    def instance_count(self) -> int:
        return len(self._vertex_indices)

    def project(self, solve_positions) -> None:
        # target position은 항상 고정. 갱신할 projection이 없다.
        pass

    def add_lhs(self, assembler: MatrixAssembler) -> None:
        for vertex_index in self._vertex_indices:
            index = int(vertex_index)
            assembler.add(index, index, self._weight)

    def add_rhs(self, system_rhs: np.ndarray) -> None:
        np.add.at(
            system_rhs, self._vertex_indices, self._weight * self._target_positions
        )
