"""
Adapted from https://github.com/jkiesele/caloGraphNN/blob/6d1127d807bc0dbaefcf1ed804d626272f002404/caloGraphNN_keras.py
"""

# pyright: reportMissingImports=false
import warnings

from qkeras import QDense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .utils import pairwise_concatenate


def global_exchange(x, n_vertices, n_features, prefix: str = "global_exchange"):
    """
    Compute statistics (mean, min, max) over features for all vertices in the batch
    and concatenate them to each vertex's features.

    Input shape:  (B, V, F)
    Output shape: (B, V, 4F)  # [mean, min, max, original]
    """

    mean_feat = layers.GlobalAveragePooling1D(keepdims=False, name=f"{prefix}_mean")(
        x
    )  # (B, F)
    max_feat = layers.GlobalMaxPooling1D(keepdims=False, name=f"{prefix}_max")(
        x
    )  # (B, F)
    neg = layers.Dense(
        n_features,
        use_bias=False,
        activation=None,
        kernel_initializer=keras.initializers.Constant(-1.0 * tf.eye(n_features)),
        trainable=False,
    )  # negation layer
    min_feat = neg(
        layers.GlobalMaxPooling1D(keepdims=False, name=f"{prefix}_min")(neg(x))
    )  # (B, F)

    stats = pairwise_concatenate(
        [mean_feat, min_feat, max_feat],
        axis=-1,
        name_prefix=f"{prefix}_global_stats_concat",
    )  # (B, 3F)
    stats_expanded = layers.Reshape(
        (1, 3 * n_features), name=f"{prefix}_global_stats_reshape"
    )(stats)  # (B, 1, 3F)
    repeated = layers.UpSampling1D(
        size=n_vertices, name=f"{prefix}_global_stats_upsample"
    )(stats_expanded)
    out = layers.Concatenate(axis=-1, name=f"{prefix}_concat")(
        [repeated, x]
    )  # (B, V, 4F)

    return out


class GravNetCore(keras.layers.Layer):
    """
    GravNet neighbour-aggregation for nearest n_neighbours in learned coordinate space.

    Inputs:
        coords: (B, V, S)
        feats:  (B, V, F_prop)
    Returns:
        aggregated features (B, V, 2*F_prop) -> concat([fmax, fmean])
    """

    def __init__(self, n_neighbours: int, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_neighbours = n_neighbours

    def call(self, coords, feats):
        """
        coords: (B, V, S)
        feats:  (B, V, F_prop)
        returns: aggregated features (B, V, 2*F_prop) -> concat([fmax, fmean])
        """
        B = tf.shape(feats)[0]
        V = tf.shape(feats)[1]

        # squared distances (B, V, V)
        dist = self._euclidean_squared(coords, coords)

        dist = (
            dist + tf.repeat(tf.expand_dims(tf.eye(V, V), axis=0), B, axis=0) * 1e9
        )  # ? or better to do ranked_indices[:, :, 1:]
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
    def _euclidean_squared(A, B):
        sub = -2.0 * tf.matmul(A, tf.transpose(B, [0, 2, 1]))
        dotA = tf.reduce_sum(tf.square(A), axis=2, keepdims=True)
        dotB = tf.reduce_sum(tf.square(B), axis=2, keepdims=True)
        dotB = tf.transpose(dotB, [0, 2, 1])
        return sub + dotA + dotB


class GravNetLayer(keras.layers.Layer):
    """
    GravNet layer that accepts pre-constructed transform layers.

    Input:  x of shape (B, V, F_in)
    Output: y of shape (B, V, n_filters)   (optionally also coordinates)
    """

    def __init__(
        self,
        input_feature_transform,
        input_spatial_transform,
        output_feature_transform,
        n_neighbours,
        n_dimensions,
        also_coordinates=False,
        feature_dropout=-1.0,
        fix_coordinate_space=False,
        masked_coordinate_offset=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.input_feature_transform = input_feature_transform
        self.input_spatial_transform = input_spatial_transform
        self.output_feature_transform = output_feature_transform

        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.also_coordinates = also_coordinates
        self.feature_dropout = feature_dropout
        self.fix_coordinate_space = fix_coordinate_space
        self.masked_coordinate_offset = masked_coordinate_offset

        self.dropout = None
        if 0.0 < feature_dropout < 1.0:
            self.dropout = keras.layers.Dropout(feature_dropout)

        self.core = GravNetCore(
            self.n_neighbours, name=(name + "_core") if name else None
        )

    def call(self, x, training=False):
        if self.masked_coordinate_offset is not None:
            raise NotImplementedError("masked_coordinate_offset not implemented")

        # F_in -> F_prop
        fprop = self.input_feature_transform(x)
        if self.dropout is not None:
            fprop = self.dropout(fprop, training=training)

        # F_in -> coords (S) or slice from x
        if self.input_spatial_transform is not None:
            coords = self.input_spatial_transform(x)
        else:
            coords = x[:, :, : self.n_dimensions]

        # neighbour aggregation
        neigh = self.core(coords, fprop)

        merged = tf.concat([x, neigh], axis=-1)

        out = self.output_feature_transform(merged)

        if self.also_coordinates:
            return [out, coords]
        return out


class QGravNetLayer(keras.layers.Layer):
    """
    GravNetLayer wrapper that builds internal transforms with quantized layers.

    Input:  x of shape (B, V, F_in)
    Output: y of shape (B, V, n_filters)   (optionally also coordinates)
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
        input_feature_transform = QDense(
            n_propagate,
            kernel_initializer=other_kernel_initializer,
            kernel_quantizer=feature_kernel_quantizer,
            bias_quantizer=feature_bias_quantizer,
            activation=feature_activation,
        )

        # F_in → S (latent coordinate space)
        if not fix_coordinate_space:
            input_spatial_transform = QDense(
                n_dimensions,
                kernel_initializer=coordinate_kernel_initializer,
                kernel_quantizer=coordinate_kernel_quantizer,
                bias_quantizer=coordinate_bias_quantizer,
                activation=coordinate_activation,
            )
        else:
            input_spatial_transform = None
            # FIXME does this make sense for multiple blocks?
            warnings.warn(
                "QGravNetLayer.fix_coordinate_space=True currently slices "
                "the first n_dimensions from input features as coordinates as in the original"
                "GravNet implementation. This may not be appropriate for multiple stacked blocks.",
                UserWarning,
            )

        # concat([x, aggregated]) → F_out
        output_feature_transform = QDense(
            n_filters,
            kernel_initializer=other_kernel_initializer,
            kernel_quantizer=output_kernel_quantizer,
            bias_quantizer=output_bias_quantizer,
            activation=output_activation,
        )

        self.gravnet = GravNetLayer(
            input_feature_transform=input_feature_transform,
            input_spatial_transform=input_spatial_transform,
            output_feature_transform=output_feature_transform,
            n_neighbours=n_neighbours,
            n_dimensions=n_dimensions,
            also_coordinates=also_coordinates,
            feature_dropout=feature_dropout,
            fix_coordinate_space=fix_coordinate_space,
            masked_coordinate_offset=masked_coordinate_offset,
            name=name + "_composite" if name is not None else None,
        )

    def call(self, x, training=False):
        return self.gravnet(x, training=training)
