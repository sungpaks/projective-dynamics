"""Local Step의 projection을 모아 Lq=b 구성하고, 풀어서 solve_positions를 갱신"""

import taichi as ti

from cloth.constraint import ProjectiveConstraintSet
from cloth.matrix_assembler import DiagonalMatrixAssembler
from lib.taichi_typing import TaichiTemplate


@ti.data_oriented
class DiagonalGlobalSystem:
    def __init__(
        self,
        vertex_count: int,
        time_step: float,
        constraints: list[ProjectiveConstraintSet],
    ) -> None:
        self._vertex_count = vertex_count
        self._time_step = time_step
        self._constraints = tuple(constraints)

        # LHS: diagonal only.     L in R^{NxN}
        self.lhs_diagonal = ti.field(dtype=ti.f32, shape=vertex_count)
        self._build_lhs()

        # RHS: 3차원 벡터.          q, b in R^{Nx3}
        self.system_rhs = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)

    def _build_lhs(self) -> None:
        """Global LHS를 사전 빌드"""
        assembler = DiagonalMatrixAssembler(self._vertex_count)

        self._add_inertial_lhs(assembler)

        for constraint in self._constraints:
            constraint.add_lhs(assembler)

        self.lhs_diagonal.from_numpy(assembler.values)

    def _add_inertial_lhs(self, assembler: DiagonalMatrixAssembler) -> None:
        """질량항 contribution을 global LHS에 추가"""
        for vertex_index in range(self._vertex_count):
            assembler.add(vertex_index, vertex_index, 1.0 / (self._time_step**2))

    def solve(
        self,
        predicted_positions,
        solve_positions,
    ) -> None:
        self._initialize_rhs(predicted_positions)

        for constraint in self._constraints:
            constraint.add_rhs(self.system_rhs)

        self._solve_linear_system(solve_positions)

    @ti.kernel
    def _initialize_rhs(self, predicted_positions: TaichiTemplate):
        """Global RHS를 초기화"""
        for vertex_index in range(self._vertex_count):
            self.system_rhs[vertex_index] = predicted_positions[vertex_index] / (
                self._time_step**2
            )

    @ti.kernel
    def _solve_linear_system(self, solve_positions: TaichiTemplate):
        for vertex_index in range(self._vertex_count):
            solve_positions[vertex_index] = (
                self.system_rhs[vertex_index] / self.lhs_diagonal[vertex_index]
            )
