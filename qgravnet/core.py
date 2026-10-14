# pyright: reportMissingImports=false
from typing import Literal

import tensorflow as tf
from tensorflow import keras

from .selectors import FullSelector, NeighbourSelector


@keras.utils.register_keras_serializable(package="qgravnet")
class GravNetCore(keras.layers.Layer):
    """
    GravNet neighbour-aggregation for nearest n_neighbours in learned coordinate space.

    Inputs:
        coords: (B, V, S)
        feats:  (B, V, F_prop)
    Returns:
        aggregated features (B, V, 2*F_prop) -> concat([fmax, fmean])
        with global_stats=True: (B, V, 5*F_prop) -> concat([fmax, fmean, gmean, gmin, gmax])

    global_stats appends mean/min/max of feats over all vertices (a fused GlobalExchange).
    In HLS these are gathered while the core buffers the event, so they add no full-V barrier.
    """

    def __init__(
        self,
        n_neighbours: int,
        distance_metric: Literal["l1", "l2_squared"] = "l1",
        distance_scale: float = 10.0,
        selector: NeighbourSelector | None = None,
        global_stats: bool = False,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.n_neighbours = n_neighbours
        self.global_stats = global_stats
        self.distance_metric = distance_metric
        self.distance_scale = distance_scale
        self.selector = selector or FullSelector()

        distance_fns = {
            "l1": self._l1_distance,
            "l2_squared": self._l2_squared_distance,
        }

        if self.distance_metric not in distance_fns:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")

        self._distance_fn = distance_fns[self.distance_metric]

    def call(self, inputs):
        """
        coords: (B, V, S)
        feats:  (B, V, F_prop)
        returns: aggregated features (B, V, 2*F_prop) -> concat([fmax, fmean])
        """
        coords, feats = inputs

        # squared distances (B, V, V)
        dist = self._distance_fn(coords, coords)

        dist = self._mask_self(dist)

        # restict neighbour search (approximate kNN)
        dist = self.selector.restrict(dist, coords)

        ranked_distances, ranked_indices = tf.math.top_k(
            -dist, k=self.n_neighbours, sorted=True
        )
        ranked_distances = -ranked_distances

        neigh_feats = tf.gather(
            feats, ranked_indices, axis=1, batch_dims=1
        )  # (B, V, k, F_prop)

        w = tf.exp(-self.distance_scale * ranked_distances)
        w = tf.expand_dims(w, -1)
        weighted = neigh_feats * w

        fmax = tf.reduce_max(weighted, axis=2)
        fmean = tf.reduce_mean(weighted, axis=2)
        if not self.global_stats:
            return tf.concat([fmax, fmean], axis=-1)

        # per-event stats of the propagated features, same order as GlobalExchange
        gstats = tf.concat(
            [
                tf.reduce_mean(feats, axis=1, keepdims=True),
                tf.reduce_min(feats, axis=1, keepdims=True),
                tf.reduce_max(feats, axis=1, keepdims=True),
            ],
            axis=-1,
        )  # (B, 1, 3F_prop)
        gstats = tf.tile(gstats, [1, tf.shape(feats)[1], 1])
        return tf.concat([fmax, fmean, gstats], axis=-1)

    def compute_output_shape(self, input_shape):
        coords_shape, feats_shape = input_shape
        B, V, F = feats_shape
        return tf.TensorShape([B, V, (5 if self.global_stats else 2) * F])

    @staticmethod
    def _mask_self(dist):
        """Set self-distances to a large value to exclude self from nearest neighbours."""
        shape = tf.shape(dist)
        V = shape[1]

        eye = tf.eye(V, batch_shape=shape[0:1], dtype=dist.dtype)
        large = tf.constant(1e9, dtype=dist.dtype)

        return dist + eye * large

    @staticmethod
    def _l2_squared_distance(A, B):
        sub = -2.0 * tf.matmul(A, tf.transpose(B, [0, 2, 1]))
        dotA = tf.reduce_sum(tf.square(A), axis=2, keepdims=True)
        dotB = tf.reduce_sum(tf.square(B), axis=2, keepdims=True)
        dotB = tf.transpose(dotB, [0, 2, 1])
        return sub + dotA + dotB

    @staticmethod
    def _l1_distance(A, B):
        return tf.reduce_sum(tf.abs(A[:, :, None, :] - B[:, None, :, :]), axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "n_neighbours": self.n_neighbours,
                "distance_metric": self.distance_metric,
                "distance_scale": self.distance_scale,
                "global_stats": self.global_stats,
                "selector": keras.utils.serialize_keras_object(self.selector),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["selector"] = keras.utils.deserialize_keras_object(config["selector"])
        return cls(**config)
