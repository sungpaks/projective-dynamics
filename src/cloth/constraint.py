# pyright: reportInvalidTypeForm=false

from typing import Protocol

import numpy as np
import taichi as ti


class ProjectiveConstraintSet(Protocol):
    @property
    def instance_count(self) -> int: ...

    def project(self, solve_positions) -> None:
        """Local Step에서 p_i 갱신 (per constraint-instance)"""

    def add_lhs(self, assembler) -> None:
        """고정된 Global Matrix contribution 추가"""

    def add_rhs(self, global_rhs) -> None:
        """현재 p_i로 global RHS contribution 추가"""


@ti.data_oriented
class IdentityConstraintSet:
    """
    더미 ConstraintSet.
    - selection은 각 정점별로
    -
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

    def add_lhs(self, assembler) -> None:
        # 정점 하나씩 선택되므로 global L matrix의 대각성분 하나에만 영향(weight)을 준다
        self._add_lhs(assembler)

    def add_rhs(self, global_rhs) -> None:
        # 정점 하나씩 선택되므로 global rhs의 정점별 항 하나에만 영향(weight)을 준다.
        for instance_index in self.vertex_indices:
            vertex_index = self.vertex_indices[instance_index]
            global_rhs[vertex_index] += self._weight * self.projections[instance_index]

    @ti.kernel
    def _project_all(self, solve_positions: ti.template()):
        for instance_index in self.vertex_indices:
            # instance_index: constraint instance의 번호
            # vertex_index: instance가 선택한 정점 번호
            # EmptyConstraint는 정점 하나만 선택하므로 instance_index == vertex_index
            vertex_index = self.vertex_indices[instance_index]
            self.projections[instance_index] = solve_positions[vertex_index]

    @ti.kernel
    def _add_lhs(self, assembler: ti.template()):
        for vertex_index in self.vertex_indices:
            assembler.add(vertex_index, vertex_index, self._weight)
