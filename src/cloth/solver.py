"""천의 물리적 상태 및 시간 적분 등"""

import numpy as np
import taichi as ti

@ti.data_oriented
class ClothSolver:
  def __init__(self, initial_positions: np.ndarray):
    vertex_count = len(initial_positions)

    self.positions = ti.Vector.field(
      3,
      dtype=ti.f32,
      shape=vertex_count
    )
    self.velocities = ti.Vector.field(
      3,
      dtype=ti.f32,
      shape=vertex_count
    )
    self.predicted_positions = ti.Vector.field(
      3,
      dtype=ti.f32,
      shape=vertex_count
    )

    self.positions.from_numpy(initial_positions)

  def step(
      self,
      time_step: float,
      gravity: tuple[float, float, float]
  ) -> None:
    # Python 코드에서 한 프레임마다 호출, 물리 연산 algorithm 진행
    pass

# pyright: reportInvalidTypeForm=false
  @ti.kernel
  def _predict_positions(
    self,
    time_step: ti.f32,
    gravity: ti.types.vector(3, ti.f32),
  ):
    # 모든 정점의 예상 위치 계산
    pass