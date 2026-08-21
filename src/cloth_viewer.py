"""Render a static triangle grid with Taichi GGUI."""

import numpy as np
import taichi as ti


# The grid has GRID_SIZE vertices along each side.
GRID_SIZE = 10
WINDOW_RESOLUTION = (960, 720)


def create_cloth_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Create vertices and triangle indices for a square cloth grid."""
    vertex_count = size * size
    cell_count = (size - 1) * (size - 1)

    # Each vertex stores an (x, y, z) position using 32-bit floats.
    vertices = np.zeros((vertex_count, 3), dtype=np.float32)

    for row in range(size):
        for column in range(size):
            vertex_index = row * size + column

            # Map grid coordinates to [-1, 1] and place the cloth on the x-z plane.
            x = column / (size - 1) * 2.0 - 1.0
            z = row / (size - 1) * 2.0 - 1.0
            vertices[vertex_index] = (x, 0.0, z)

    # One square cell becomes two triangles, with three indices per triangle.
    indices = np.zeros(cell_count * 2 * 3, dtype=np.int32)

    write_index = 0
    for row in range(size - 1):
        for column in range(size - 1):
            top_left = row * size + column
            top_right = top_left + 1
            bottom_left = (row + 1) * size + column
            bottom_right = bottom_left + 1

            indices[write_index : write_index + 6] = (
                top_left,
                bottom_left,
                top_right,
                top_right,
                bottom_left,
                bottom_right,
            )
            write_index += 6

    return vertices, indices


def main() -> None:
    # Metal is Apple's GPU backend. Taichi compiles kernels and rendering for it.
    ti.init(arch=ti.metal)

    vertices_np, indices_np = create_cloth_grid(GRID_SIZE)

    # Taichi fields are GPU-accessible buffers, similar to WebGPU GPUBuffer objects.
    vertices = ti.Vector.field(3, dtype=ti.f32, shape=len(vertices_np))
    indices = ti.field(dtype=ti.i32, shape=len(indices_np))

    # Copy the mesh created on the CPU by NumPy into Taichi-managed memory.
    vertices.from_numpy(vertices_np)
    indices.from_numpy(indices_np)

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

        scene.ambient_light((0.45, 0.45, 0.45))
        scene.point_light(pos=(2.0, 3.0, 2.0), color=(1.0, 1.0, 1.0))
        scene.mesh(
            vertices,
            indices=indices,
            color=(0.25, 0.65, 0.85),
            two_sided=True,
            show_wireframe=True,
        )

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
