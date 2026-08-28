"""Reusable RGB axis visualization for Taichi GGUI scenes."""

import taichi as ti


class AxisHelper:
    """Render X, Y, and Z axes from an origin with configurable extents."""

    def __init__(
        self,
        *,
        positive_length: float,
        negative_length: float = 0.0,
        width: float = 2.0,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        colors: tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ] = ((1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (0.1, 0.3, 1.0)),
    ) -> None:
        self.width = width
        self.colors = colors
        self.axes = tuple(
            self._create_axis(origin, axis_index, negative_length, positive_length)
            for axis_index in range(3)
        )

    @staticmethod
    def _create_axis(
        origin: tuple[float, float, float],
        axis_index: int,
        negative_length: float,
        positive_length: float,
    ):
        start = list(origin)
        end = list(origin)
        start[axis_index] -= negative_length
        end[axis_index] += positive_length

        axis = ti.Vector.field(3, dtype=ti.f32, shape=2)
        axis[0] = start
        axis[1] = end
        return axis

    def draw(self, scene) -> None:
        """Add all three colored axes to the current scene frame."""
        for axis, color in zip(self.axes, self.colors):
            scene.lines(axis, width=self.width, color=color)
