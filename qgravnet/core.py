from typing import Literal

import tensorflow as tf
from tensorflow import keras


class GravNetCore(keras.layers.Layer):
    """
    GravNet neighbour-aggregation for nearest n_neighbours in learned coordinate space.

    Inputs:
        coords: (B, V, S)
        feats:  (B, V, F_prop)
    Returns:
        aggregated features (B, V, 2*F_prop) -> concat([fmax, fmean])
    """
    def __init__(
        self, 
        n_neighbours: int, 
        distance_metric: Literal["l1", "l2_squared"] = "l1", 
        name: str | None = None, 
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.n_neighbours = n_neighbours
        self.distance_metric = distance_metric

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
        B = tf.shape(feats)[0]
        V = tf.shape(feats)[1]

        # squared distances (B, V, V)
        dist = self._distance_fn(coords, coords)

        dist = self._mask_self(dist)

        ranked_distances, ranked_indices = tf.math.top_k(
            -dist, k=self.n_neighbours, sorted=True
        )
        ranked_distances = -ranked_distances

        neigh_feats = tf.gather(
            feats, ranked_indices, axis=1, batch_dims=1
        )  # (B, V, k, F_prop)

        w = tf.exp(-10.0 * ranked_distances)
        w = tf.expand_dims(w, -1)
        weighted = neigh_feats * w

        fmax = tf.reduce_max(weighted, axis=2)
        fmean = tf.reduce_mean(weighted, axis=2)
        return tf.concat([fmax, fmean], axis=-1)

    @staticmethod
    def _mask_self(dist):
        """Set self-distances to a large value to exclude self from nearest neighbours."""
        large = tf.constant(1e9, dtype=dist.dtype)
        shape = tf.shape(dist)
        B = shape[0]
        V = shape[1]
        return tf.linalg.set_diag(dist, tf.fill([B, V], large))

    @staticmethod
    def _l2_squared_distance(A, B):
        sub = -2.0 * tf.matmul(A, tf.transpose(B, [0, 2, 1]))
        dotA = tf.reduce_sum(tf.square(A), axis=2, keepdims=True)
        dotB = tf.reduce_sum(tf.square(B), axis=2, keepdims=True)
        dotB = tf.transpose(dotB, [0, 2, 1])
        return sub + dotA + dotB

    @staticmethod
    def _l1_distance(A, B):
        return tf.reduce_sum(
            tf.abs(A[:, :, None, :] - B[:, None, :, :]),
            axis=-1
        )
    
    def get_config(self):
        config = super().get_config()
        config.update({'n_neighbours': self.n_neighbours, 'distance_metric': self.distance_metric})
        return config