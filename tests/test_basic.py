# pyright: reportMissingImports=false

import numpy as np
import pytest
import tensorflow as tf


def test_gravnetcore_neighbors():
    """Ensure GravNetCore neighbor aggregation returns correct shape and no NaNs."""
    from qgravnet.core import GravNetCore

    core = GravNetCore(n_neighbours=5, name="core")
    coords = tf.random.normal((1, 20, 4))
    feats = tf.random.normal((1, 20, 6))

    out = core([coords, feats])

    # output = concat([fmax, fmean]) -> shape (B, V, 2 * F_prop)
    assert out.shape == (1, 20, 12)
    assert not bool(tf.reduce_any(tf.math.is_nan(out)))


def test_qgravnetlayer_forward():
    """Forward-pass test for QGravNetLayer."""
    from qgravnet.layers import QGravNetLayer

    layer = QGravNetLayer(
        n_neighbours=4,
        n_dimensions=3,
        n_filters=8,
        n_propagate=6,
        name="test_layer",
    )

    x = tf.random.normal((2, 10, 5))
    y = layer(x)

    assert y.shape == (2, 10, 8)
    assert not bool(tf.reduce_any(tf.math.is_nan(y)))


@pytest.mark.parametrize("factory_cls", ["GravNetFactory", "QGravNetFactory"])
def test_factory_model_dual_head(factory_cls):
    """Test that both factory variants build a dual-head model and run a forward pass."""
    from qgravnet import factory as factory_module

    factory = getattr(factory_module, factory_cls)(
        n_blocks=2, n_neighbours=4, output_head="dual"
    )
    model = factory.create_keras_model(n_vertices=16, n_features=8)

    x = tf.random.normal((1, 16, 8))
    energies, classes = model(x)

    assert energies.shape == (1, 1)
    assert classes.shape == (1, 1)

    assert not bool(tf.reduce_any(tf.math.is_nan(energies)))
    assert not bool(tf.reduce_any(tf.math.is_nan(classes)))


@pytest.mark.parametrize("factory_cls", ["GravNetFactory", "QGravNetFactory"])
def test_factory_model_oc_head(factory_cls):
    """Test that both factory variants build an OC-style model and run a forward pass."""
    from qgravnet import factory as factory_module

    factory = getattr(factory_module, factory_cls)(
        n_blocks=2,
        n_neighbours=4,
        output_head="oc",
        output_dim=5,  # test non-default output size
    )
    model = factory.create_keras_model(n_vertices=16, n_features=8)

    x = tf.random.normal((1, 16, 8))
    out = model(x)

    # Shape should be (B, V, output_dim)
    assert out.shape == (1, 16, 5)
    assert not bool(tf.reduce_any(tf.math.is_nan(out)))


def test_binned_selector_restrict_example_input():
    """BinnedSelector should keep only in-bin neighbours for a simple example input."""
    from qgravnet.selectors import BinnedSelector

    selector = BinnedSelector(bins_per_axis=4, window=0, clip_min=-1.0, clip_max=1.0)

    # One batch, four vertices, one coordinate dimension
    coords = tf.constant([[[-0.9], [-0.8], [0.1], [0.9]]], dtype=tf.float32)
    # Use true pairwise L2 distances computed from coords.
    delta = coords[:, :, None, :] - coords[:, None, :, :]
    dist = tf.norm(delta, axis=-1)

    restricted = selector.restrict(dist, coords).numpy()

    large = 1e9
    expected = [
        [
            [0.0, 0.1, large, large],
            [0.1, 0.0, large, large],
            [large, large, 0.0, large],
            [large, large, large, 0.0],
        ]
    ]

    np.testing.assert_allclose(restricted, expected, atol=1e-6)
