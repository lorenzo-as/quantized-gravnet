"""
Adapted from https://github.com/jkiesele/caloGraphNN/blob/6d1127d807bc0dbaefcf1ed804d626272f002404/caloGraphNN_keras.py
"""

# pyright: reportMissingImports=false
from typing import Literal
import warnings

from qkeras import QDense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

from .core import GravNetCore


class GlobalExchange(keras.layers.Layer):
    """
    Compute statistics (mean, min, max) over features for all vertices in the batch
    and concatenate them to each vertex's features.

    Input shape:  (B, V, F)
    Output shape: (B, V, 4F)  # [mean, min, max, original]
    """

    def __init__(self, vertex_mask=None, **kwargs):
        super().__init__(**kwargs)
        if vertex_mask is not None:
            raise NotImplementedError

    def call(self, x):
        # x: (B, V, F)
        if K.ndim(x) != 3:
            raise ValueError(f"GlobalExchange expects input of shape (B, V, F) but received shape {K.int_shape(x)}")

        mean = K.mean(x, axis=1, keepdims=True)  # (B, 1, F)
        vmin = K.min(x, axis=1, keepdims=True)  # (B, 1, F)
        vmax = K.max(x, axis=1, keepdims=True)  # (B, 1, F)

        stats = K.concatenate([mean, vmin, vmax], axis=-1)  # (B, 1, 3F)

        V = tf.shape(x)[1]
        stats = tf.tile(stats, [1, V, 1])  # (B, V, 3F)

        return K.concatenate([stats, x], axis=-1)  # (B, V, 4F) [mean, min, max, x]

    def compute_output_shape(
        self, input_shape
    ):  # with ops.tile, static shape info appears to be lost vs tf.tile thus this is needed
        B, V, F = input_shape
        return (B, V, 4 * F)

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
        neigh = self.core([coords, fprop])

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
