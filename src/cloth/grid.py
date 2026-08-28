"""Procedural grid mesh generation for cloth simulation."""

import numpy as np


def create_cloth_grid(
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create vertices, triangle indices, and unique edge indices."""
    vertex_count = size * size
    cell_count = (size - 1) * (size - 1)

    # Each vertex stores an (x, y, z) position using 32-bit floats.
    vertices = np.zeros((vertex_count, 3), dtype=np.float32)

    for row in range(size):
        for column in range(size):
            vertex_index = row * size + column

            # Place the cloth on the x-y plane, with the first row at the top.
            x = column / (size - 1) * 2.0 - 1.0
            y = 1.0 - row / (size - 1) * 2.0
            vertices[vertex_index] = (x, y, 0.0)

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

    # scene.lines() needs pairs of vertex indices. A set removes shared edges.
    edges: set[tuple[int, int]] = set()
    for triangle in indices.reshape(-1, 3):
        for start, end in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            start_index = int(start)
            end_index = int(end)
            edge = (
                min(start_index, end_index),
                max(start_index, end_index),
            )
            edges.add(edge)

    edge_indices = np.array(sorted(edges), dtype=np.int32).reshape(-1)

    return vertices, indices, edge_indices
