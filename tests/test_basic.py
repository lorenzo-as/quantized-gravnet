# pyright: reportMissingImports=false

import tensorflow as tf


def test_gravnetcore_neighbors():
    """Ensure GravNetCore neighbor aggregation returns correct shape and no NaNs."""
    from qgravnet.layers import GravNetCore

    core = GravNetCore(n_neighbours=5, name="core")
    coords = tf.random.normal((1, 20, 4))
    feats = tf.random.normal((1, 20, 6))

    out = core(coords, feats)

    # output = concat([fmax, fmean]) -> shape (B, V, 2 * F_prop)
    assert out.shape == (1, 20, 12)
    assert not tf.math.reduce_any(tf.math.is_nan(out))


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
    assert not tf.math.reduce_any(tf.math.is_nan(y))


def test_factory_model_dual_head():
    """Test that QGravNetFactory builds a dual-head model and runs a forward pass."""
    from qgravnet.factory import QGravNetFactory

    factory = QGravNetFactory(n_blocks=2, n_neighbours=4, output_head="dual")
    model = factory.create_keras_model(n_vertices=16, n_features=8)

    x = tf.random.normal((1, 16, 8))
    energies, classes = model(x)

    assert energies.shape == (1, 1)
    assert classes.shape == (1, 1)


def test_factory_model_oc_head():
    """Test that QGravNetFactory builds an OC-style model and runs a forward pass."""
    from qgravnet.factory import QGravNetFactory

    factory = QGravNetFactory(
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
    assert not tf.math.reduce_any(tf.math.is_nan(out))
