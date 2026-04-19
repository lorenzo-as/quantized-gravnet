# pyright: reportMissingImports=false

import pytest
import tensorflow as tf


def _assert_outputs_close(y_ref, y_loaded, atol=1e-5):
    if isinstance(y_ref, (list, tuple)):
        assert isinstance(y_loaded, (list, tuple))
        assert len(y_ref) == len(y_loaded)
        for ref, loaded in zip(y_ref, y_loaded):
            tf.debugging.assert_near(ref, loaded, atol=atol)
        return

    tf.debugging.assert_near(y_ref, y_loaded, atol=atol)


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


def test_gravnetcore_l1_exact_nearest_neighbor_aggregation():
    """Check exact weighted aggregation for a tiny configuration with unique nearest neighbours."""
    from qgravnet.core import GravNetCore

    core = GravNetCore(n_neighbours=1, distance_metric="l1", name="core_l1_exact")

    coords = tf.constant([[[0.0], [1.0], [3.0]]], dtype=tf.float32)
    feats = tf.constant([[[10.0], [20.0], [30.0]]], dtype=tf.float32)

    out = core([coords, feats])

    w_01 = float(tf.exp(tf.constant(-10.0, dtype=tf.float32)).numpy())
    w_21 = float(tf.exp(tf.constant(-20.0, dtype=tf.float32)).numpy())
    expected = tf.constant(
        [
            [
                [20.0 * w_01, 20.0 * w_01],
                [10.0 * w_01, 10.0 * w_01],
                [20.0 * w_21, 20.0 * w_21],
            ]
        ],
        dtype=tf.float32,
    )

    tf.debugging.assert_near(out, expected, atol=1e-6)


def test_gravnetcore_l2_squared_exact_nearest_neighbor_aggregation():
    """Check exact weighted aggregation for the squared L2 metric."""
    from qgravnet.core import GravNetCore

    core = GravNetCore(
        n_neighbours=1, distance_metric="l2_squared", name="core_l2_exact"
    )

    coords = tf.constant([[[0.0], [2.0]]], dtype=tf.float32)
    feats = tf.constant([[[3.0], [5.0]]], dtype=tf.float32)

    out = core([coords, feats])

    weight = float(tf.exp(tf.constant(-40.0, dtype=tf.float32)).numpy())
    expected = tf.constant(
        [[[5.0 * weight, 5.0 * weight], [3.0 * weight, 3.0 * weight]]],
        dtype=tf.float32,
    )

    tf.debugging.assert_near(out, expected, atol=1e-8)


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


def test_qgravnetblock_forward():
    """Forward-pass test for QGravNetBlock."""
    from qgravnet.model import QGravNetBlock

    block = QGravNetBlock(
        n_neighbours=4,
        n_dimensions=3,
        n_filters=8,
        n_propagate=6,
        name="test_block",
    )

    x = tf.random.normal((2, 10, 5))
    y = block(x, training=False)

    assert y.shape == (2, 10, 8)
    assert not bool(tf.reduce_any(tf.math.is_nan(y)))


def test_qgravnetmodel_forward():
    """Forward-pass test for QGravNetModel."""
    from qgravnet.model import QGravNetModel

    model = QGravNetModel(
        n_blocks=2,
        n_neighbours=4,
        n_dimensions=3,
        n_filters=8,
        n_propagate=6,
        n_postgn_dense_blocks=2,
        output_dim=5,
        name="test_qgravnet_model",
    )

    x = tf.random.normal((2, 10, 5))
    y = model(x, training=False)

    assert y.shape == (2, 10, 5)
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

    _assert_outputs_close(y_ref, y_loaded, atol=1e-5)


def test_qgravnetblock_save_load_roundtrip(tmp_path):
    """Smoke test save/load roundtrip for QGravNetBlock outputs."""
    from qgravnet.model import QGravNetBlock

    inputs = tf.keras.Input(shape=(10, 5))
    outputs = QGravNetBlock(
        n_neighbours=4,
        n_dimensions=3,
        n_filters=8,
        n_propagate=6,
        name="smoke_block",
    )(inputs)
    model = tf.keras.Model(inputs, outputs)

    x = tf.random.normal((2, 10, 5))
    y_ref = model(x, training=False)

    save_path = tmp_path / "smoke_block.keras"
    model.save(save_path)

    loaded = tf.keras.models.load_model(save_path)
    y_loaded = loaded(x, training=False)

    _assert_outputs_close(y_ref, y_loaded, atol=1e-5)


def test_qgravnetmodel_save_load_roundtrip(tmp_path):
    """Smoke test save/load roundtrip for the subclassed QGravNetModel."""
    from qgravnet.model import QGravNetModel

    model = QGravNetModel(
        n_blocks=2,
        n_neighbours=4,
        n_dimensions=3,
        n_filters=8,
        n_propagate=6,
        n_postgn_dense_blocks=2,
        output_dim=5,
        name="smoke_qgravnet_model",
    )

    x = tf.random.normal((2, 10, 5))
    y_ref = model(x, training=False)

    save_path = tmp_path / "smoke_qgravnet_model.keras"
    model.save(save_path)

    loaded = tf.keras.models.load_model(save_path)
    y_loaded = loaded(x, training=False)

    _assert_outputs_close(y_ref, y_loaded, atol=1e-5)


@pytest.mark.parametrize("factory_cls", ["GravNetFactory", "QGravNetFactory"])
@pytest.mark.parametrize(
    ("output_head", "output_dim"),
    [("dual", 4), ("oc", 5)],
)
@pytest.mark.parametrize(
    ("neighbour_selector", "selector_cfg"),
    [
        ("full", None),
        (
            "binned",
            {"bins_per_axis": 4, "window": 1, "clip_min": -1.0, "clip_max": 1.0},
        ),
    ],
)
def test_factory_model_save_load_roundtrip(
    tmp_path, factory_cls, output_head, output_dim, neighbour_selector, selector_cfg
):
    """Factory-built models should survive forward-pass and save/load roundtrips."""
    from qgravnet import factory as factory_module

    factory = getattr(factory_module, factory_cls)(
        n_blocks=2,
        n_neighbours=4,
        n_dimensions=3,
        n_filters=8,
        n_propagate=6,
        n_postgn_dense_blocks=2,
        output_head=output_head,
        output_dim=output_dim,
        neighbour_selector=neighbour_selector,
        selector_cfg=selector_cfg,
    )
    model = factory.create_keras_model(n_vertices=10, n_features=5)

    x = tf.random.normal((2, 10, 5))
    y_ref = model(x, training=False)

    save_path = tmp_path / f"{factory_cls}_{output_head}_{neighbour_selector}.keras"
    model.save(save_path)

    loaded = tf.keras.models.load_model(save_path)
    y_loaded = loaded(x, training=False)

    _assert_outputs_close(y_ref, y_loaded, atol=1e-5)


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

    tf.debugging.assert_near(
        tf.constant(restricted, dtype=tf.float32),
        tf.constant(expected, dtype=tf.float32),
        atol=1e-6,
    )
