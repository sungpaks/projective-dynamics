"""Render a static triangle grid with Taichi GGUI."""

import taichi as ti

from cloth.grid import create_cloth_grid

# The grid has GRID_SIZE vertices along each side.
GRID_SIZE = 10
WINDOW_RESOLUTION = (960, 720)


def main() -> None:
    # Metal is Apple's GPU backend. Taichi compiles kernels and rendering for it.
    ti.init(arch=ti.metal)

    vertices_np, indices_np, edge_indices_np = create_cloth_grid(GRID_SIZE)

    # Taichi fields are GPU-accessible buffers, similar to WebGPU GPUBuffer objects.
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices_np))
    indices = ti.field(dtype=ti.i32, shape=len(indices_np))
    edge_indices = ti.field(dtype=ti.i32, shape=len(edge_indices_np))

    # Copy the mesh created on the CPU by NumPy into Taichi-managed memory.
    vertices.from_numpy(vertices_np)
    indices.from_numpy(indices_np)
    edge_indices.from_numpy(edge_indices_np)

    window = ti.ui.Window(
        "Projective Dynamics - Static Cloth",
        WINDOW_RESOLUTION,
        vsync=True,
    )
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Position the camera above and in front of the horizontal cloth.
    camera.position(0.0, 1.8, 2.8)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0.0, 1.0, 0.0)

    while window.running:
        # RMB rotates the camera; W/A/S/D/E/Q move it.
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        canvas.set_background_color((1.0, 1.0, 1.0))

        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(2.0, 3.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.mesh(
            vertices,
            indices=indices,
            color=(0.92, 1.0, 0.92),
            two_sided=True,
        )
        scene.lines(
            vertices,
            width=2.5,
            indices=edge_indices,
            color=(0.45, 0.85, 0.45),
        )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
