import numpy as np

from cloth.grid import create_cloth_grid


def test_create_cloth_grid_counts_and_index_bounds() -> None:
    vertices, triangles, edges = create_cloth_grid(10)

    assert vertices.shape == (100, 3)
    assert triangles.shape == (162 * 3,)
    assert edges.shape == (261 * 2,)
    assert triangles.min() == 0
    assert triangles.max() == 99


def test_create_cloth_grid_is_flat_and_centered() -> None:
    vertices, _, _ = create_cloth_grid(10)

    np.testing.assert_allclose(vertices[:, 1], 0.0)
    np.testing.assert_allclose(vertices[:, 0].min(), -1.0)
    np.testing.assert_allclose(vertices[:, 0].max(), 1.0)
    np.testing.assert_allclose(vertices[:, 2].min(), -1.0)
    np.testing.assert_allclose(vertices[:, 2].max(), 1.0)
