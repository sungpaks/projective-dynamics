"""Taichi GGUI 씬에서 재사용할 수 있는 RGB 좌표축 시각화 도구."""

from collections.abc import Sequence

import taichi as ti


class AxisHelper:
    """원점에서 시작하는 X, Y, Z축을 설정된 길이로 렌더링한다."""

    def __init__(
        self,
        *,
        positive_length: float,
        negative_length: float = 0.0,
        width: float = 2.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        directions: Sequence[Sequence[float]] = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        colors: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = ((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.3, 1.0)),
    ) -> None:
        if len(directions) != 3 or any(len(direction) != 3 for direction in directions):
            raise ValueError("directions must contain three 3D vectors")

        self.width = width
        self.colors = colors
        self.axes = tuple(
            self._create_axis(origin, direction, negative_length, positive_length)
            for direction in directions
        )

    @staticmethod
    def _create_axis(
        origin: tuple[float, float, float],
        direction: Sequence[float],
        negative_length: float,
        positive_length: float,
    ):
        start = [
            coordinate - negative_length * axis_direction
            for coordinate, axis_direction in zip(origin, direction)
        ]
        end = [
            coordinate + positive_length * axis_direction
            for coordinate, axis_direction in zip(origin, direction)
        ]

        axis = ti.Vector.field(3, dtype=ti.f32, shape=2)
        axis[0] = start
        axis[1] = end
        return axis

    def draw(self, scene) -> None:
        """색상이 지정된 세 좌표축을 현재 씬 프레임에 추가한다."""
        for axis, color in zip(self.axes, self.colors):
            scene.lines(axis, width=self.width, color=color)
