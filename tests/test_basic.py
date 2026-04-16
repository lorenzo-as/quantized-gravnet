# pyright: reportMissingImports=false

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


def test_smoke_save_load_roundtrip(tmp_path):
    """Smoke test: instantiate layer, run data, save model, load model, run inference."""
    from qgravnet.layers import QGravNetLayer

    inputs = tf.keras.Input(shape=(10, 5))
    outputs = QGravNetLayer(
        n_neighbours=4,
        n_dimensions=3,
        n_filters=8,
        n_propagate=6,
        name="smoke_layer",
    )(inputs)
    model = tf.keras.Model(inputs, outputs)

    x = tf.random.normal((2, 10, 5))
    y_ref = model(x)

    save_path = tmp_path / "smoke_model.keras"
    model.save(save_path)

    loaded = tf.keras.models.load_model(save_path)
    y_loaded = loaded(x)

    tf.debugging.assert_near(y_ref, y_loaded, atol=1e-5)
