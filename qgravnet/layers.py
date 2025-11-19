# pyright: reportMissingImports=false
import warnings

import keras
from qkeras import QDense
import tensorflow as tf


class GlobalExchange(keras.layers.Layer):
    def __init__(self, vertex_mask=None, **kwargs):
        super().__init__(**kwargs)
        if vertex_mask is not None:
            raise NotImplementedError

    def call(self, x):
        # x: (B, V, F)
        tf.debugging.assert_rank(
            x, 3, message="GlobalExchange expects input of shape (B, V, F)"
        )

        mean = tf.reduce_mean(x, axis=1, keepdims=True)  # (B, 1, F)
        vmin = tf.reduce_min(x, axis=1, keepdims=True)  # (B, 1, F)
        vmax = tf.reduce_max(x, axis=1, keepdims=True)  # (B, 1, F)

        stats = tf.concat([mean, vmin, vmax], axis=-1)  # (B, 1, 3F)

        V = tf.shape(x)[1]
        stats = tf.tile(stats, [1, V, 1])  # (B, V, 3F)

        return tf.concat([stats, x], axis=-1)  # (B, V, 4F) [mean, min, max, x]


class QGravNetLayer(keras.layers.Layer):
    """
    Input:  x of shape (B, V, F_in)
    Output: y of shape (B, V, n_filters)   (optionally also coordinates)

    Adapted from https://github.com/jkiesele/caloGraphNN/blob/6d1127d807bc0dbaefcf1ed804d626272f002404/caloGraphNN_keras.py
    """

    def __init__(
        self,
        n_neighbours,
        n_dimensions,
        n_filters,
        n_propagate,
        name,
        also_coordinates=False,
        feature_dropout=-1.0,
        coordinate_kernel_initializer=keras.initializers.Orthogonal(),
        other_kernel_initializer="glorot_uniform",
        fix_coordinate_space=False,
        masked_coordinate_offset=None,
        feature_kernel_quantizer=None,
        feature_bias_quantizer=None,
        feature_activation=None,
        coordinate_kernel_quantizer=None,
        coordinate_bias_quantizer=None,
        coordinate_activation=None,
        output_kernel_quantizer=None,
        output_bias_quantizer=None,
        output_activation="tanh",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.fix_coordinate_space = fix_coordinate_space
        self.masked_coordinate_offset = masked_coordinate_offset

        self.coordinate_kernel_initializer = coordinate_kernel_initializer
        self.other_kernel_initializer = other_kernel_initializer

        # F_in → F_prop
        self.input_feature_transform = QDense(
            n_propagate,
            kernel_initializer=other_kernel_initializer,
            kernel_quantizer=feature_kernel_quantizer,
            bias_quantizer=feature_bias_quantizer,
            activation=feature_activation,
        )

        # F_in → S (latent coordinate space)
        if not fix_coordinate_space:
            self.input_spatial_transform = QDense(
                n_dimensions,
                kernel_initializer=coordinate_kernel_initializer,
                kernel_quantizer=coordinate_kernel_quantizer,
                bias_quantizer=coordinate_bias_quantizer,
                activation=coordinate_activation,
            )
        else:
            self.input_spatial_transform = None
            # FIXME does this make sense for multiple blocks
            warnings.warn(
                "QGravNetLayer.fix_coordinate_space=True currently slices "
                "the first n_dimensions from input features as coordinates as in the original"
                "GravNet implementation. This may not be appropriate for multiple stacked blocks.",
                UserWarning,
            )

        # concat([x, aggregated]) → F_out
        self.output_feature_transform = QDense(
            n_filters,
            kernel_initializer=other_kernel_initializer,
            kernel_quantizer=output_kernel_quantizer,
            bias_quantizer=output_bias_quantizer,
            activation=output_activation,
        )

        self.dropout = None
        if 0.0 < feature_dropout < 1.0:
            self.dropout = keras.layers.Dropout(feature_dropout)

    def call(self, x, training=False):
        if self.masked_coordinate_offset is not None:
            raise NotImplementedError("masked_coordinate_offset not implemented")

        fprop = self.input_feature_transform(x)
        if self.dropout is not None:
            fprop = self.dropout(fprop, training=training)

        if self.input_spatial_transform is not None:
            coords = self.input_spatial_transform(x)
        else:
            coords = x[:, :, : self.n_dimensions]

        neigh = self.collect_neighbours(coords, fprop)

        merged = tf.concat([x, neigh], axis=-1)

        out = self.output_feature_transform(merged)

        if self.also_coordinates:
            return [out, coords]
        return out

    def collect_neighbours(self, coords, feats):
        B = tf.shape(feats)[0]
        V = tf.shape(feats)[1]

        # squared distances (B, V, V)
        dist = self._euclidean_squared(coords, coords)

        ## top-k nearest
        # ranked_distances, ranked_indices = tf.nn.top_k(-dist, k=self.n_neighbours+1)
        # ranked_indices = ranked_indices[:, :, 1:] # exclude self
        # ranked_distances = -ranked_distances[:, :, 1:]

        #! possibly ties in distances with quantization? is the below better?
        dist = dist + tf.eye(V, batch_shape=[B]) * 1e9  # mask diagonal
        ranked_distances, ranked_indices = tf.nn.top_k(-dist, k=self.n_neighbours)
        ranked_distances = -ranked_distances

        bcoord = tf.range(B)
        bcoord = tf.reshape(bcoord, (B, 1, 1))
        bcoord = tf.tile(bcoord, [1, V, self.n_neighbours])
        gather_idx = tf.stack([bcoord, ranked_indices], axis=-1)  # (B, V, k-1, 2)
        neigh_feats = tf.gather_nd(feats, gather_idx)

        w = tf.exp(-10.0 * ranked_distances)
        w = tf.expand_dims(w, -1)

        weighted = neigh_feats * w

        fmax = tf.reduce_max(weighted, axis=2)
        fmean = tf.reduce_mean(weighted, axis=2)

        return tf.concat([fmax, fmean], axis=-1)

    @staticmethod
    def _euclidean_squared(A, B):
        sub = -2.0 * tf.matmul(A, B, transpose_b=True)
        dotA = tf.reduce_sum(tf.square(A), axis=2, keepdims=True)
        dotB = tf.reduce_sum(tf.square(B), axis=2, keepdims=True)
        dotB = tf.transpose(dotB, [0, 2, 1])
        return sub + dotA + dotB
