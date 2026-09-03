"""Taichi GGUI로 정적인 삼각형 격자를 렌더링한다."""

import numpy as np
import taichi as ti

from cloth.grid import create_cloth_basis, create_cloth_grid
from cloth.rest_state import create_triangle_rest_state
from cloth.solver import ClothSolver
from lib.axis_helper import AxisHelper
from lib.time_stepper import TimeStepper

# 격자의 각 변에 정점이 GRID_SIZE개.
GRID_SIZE = 10
WINDOW_RESOLUTION = (960, 720)
CLOTH_POSITION = (0.0, 2.0, 0.0)
CLOTH_NORMAL = (0.0, 0.0, 1.0)
WORLD_AXIS_LENGTH = 100.0
OBJECT_AXIS_LENGTH = 0.3

GRAVITY = (0.0, -9.81, 0.0)
PHYSICS_STEPS_PER_SECOND = 60
PHYSICS_TIME_STEP = 1.0 / PHYSICS_STEPS_PER_SECOND


def main() -> None:
    ti.init(arch=ti.metal)

    vertices_np, indices_np, edge_indices_np = create_cloth_grid(
        GRID_SIZE,
        position=CLOTH_POSITION,
        normal=CLOTH_NORMAL,
    )
    vertices_np[:, 2] += 0.001 * np.sin(vertices_np[:, 0] * np.pi)

    fixed_vertex_indices = np.arange(
        GRID_SIZE,
        dtype=np.int32,
    )

    rest_state = create_triangle_rest_state(vertices_np, indices_np)

    solver = ClothSolver(
        vertices_np,
        rest_state,
        time_step=PHYSICS_TIME_STEP,
        fixed_vertex_indices=fixed_vertex_indices,
    )

    time_stepper = TimeStepper(
        steps_per_second=PHYSICS_STEPS_PER_SECOND,
        max_frame_time=0.25,
        max_substeps=8,
    )

    # Taichi 필드: GPU 접근 가능한 global data container. 다차원 배열.
    indices = ti.field(dtype=ti.i32, shape=len(indices_np))
    edge_indices = ti.field(dtype=ti.i32, shape=len(edge_indices_np))

    # NumPy가 CPU에서 생성한 메쉬 데이터를 Taichi 메모리로 복사.
    indices.from_numpy(indices_np)
    edge_indices.from_numpy(edge_indices_np)

    world_axes = AxisHelper(
        positive_length=WORLD_AXIS_LENGTH,
        negative_length=WORLD_AXIS_LENGTH,
        width=1.0,
        colors=((0.65, 0.35, 0.35), (0.35, 0.65, 0.35), (0.35, 0.45, 0.7)),
    )

    cloth_basis = create_cloth_basis(CLOTH_NORMAL)
    object_axes = AxisHelper(
        positive_length=OBJECT_AXIS_LENGTH,
        width=5.0,
        origin=CLOTH_POSITION,
        directions=tuple(tuple(axis) for axis in cloth_basis),
    )

    window = ti.ui.Window(
        "Projective Dynamics - Static Cloth",
        WINDOW_RESOLUTION,
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    gui = window.get_gui()
    camera = ti.ui.Camera()

    camera.position(0.2, 1, 5.0)
    camera.lookat(0.0, 1, 0.0)
    camera.up(0.0, 1.0, 0.0)
    gravity_enabled = False

    while window.running:
        gravity = GRAVITY if gravity_enabled else (0.0, 0.0, 0.0)
        time_stepper.advance(lambda time_step: solver.step(gravity))

        # 마우스 오른쪽 버튼으로 카메라를 회전하고 W/A/S/D/E/Q로 이동.
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        canvas.set_background_color((1.0, 1.0, 1.0))

        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(2.0, 3.0, 2.0), color=(1.0, 1.0, 1.0))

        world_axes.draw(scene)

        scene.mesh(
            solver.positions,
            indices=indices,
            color=(0.92, 1.0, 0.92),
            two_sided=True,
        )
        scene.lines(
            solver.positions,
            width=2.5,
            indices=edge_indices,
            color=(0.45, 0.85, 0.45),
        )

        # 천의 좌표축이 메쉬 위에 보이도록 마지막에 그린다.
        object_axes.draw(scene)

        canvas.scene(scene)

        with gui.sub_window("Controls", 0.02, 0.78, 0.16, 0.16):
            gravity_enabled = gui.checkbox("gravity", gravity_enabled)
            if gui.button("reset"):
                gravity_enabled = False
                solver.reset()
                time_stepper.reset()

        window.show()


if __name__ == "__main__":
    main()
