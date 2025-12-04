# pyright: reportMissingImports=false

from qkeras import QDense
from tensorflow import keras

from .layers import (
    GlobalExchange,
    QGravNetLayer,
)


class QGravNetBlock(keras.layers.Layer):
    def __init__(
        self,
        n_neighbours=40,
        n_dimensions=4,
        n_filters=96,
        n_propagate=22,
        block_kernel_quantizer=None,
        block_bias_quantizer=None,
        gravnet_kwargs=None,
        **kwargs,
    ):
        """
        Keras/QKeras port of the Torch hgcal-gravnet GravNetBlock.

        In Torch:
          GravNetConv -> BN -> Linear -> Tanh -> BN -> Linear -> Tanh
          -> global_exchange -> Linear -> Tanh -> BN
        """
        super().__init__(**kwargs)

        if gravnet_kwargs is None:
            gravnet_kwargs = {}

        self.n_filters = n_filters

        # GravNet core
        self.gravnet = QGravNetLayer(
            n_neighbours=n_neighbours,
            n_dimensions=n_dimensions,
            n_filters=n_filters,
            n_propagate=n_propagate,
            name=self.name + "_gravnet",
            **gravnet_kwargs,
        )

        # Post-GravNet (BN -> Dense(128,tanh) -> BN -> Dense(n_filters,tanh))
        self.bn0 = keras.layers.BatchNormalization()
        self.dense0 = QDense(
            128,
            activation="tanh",
            kernel_quantizer=block_kernel_quantizer,
            bias_quantizer=block_bias_quantizer,
        )
        self.bn1 = keras.layers.BatchNormalization()
        self.dense1 = QDense(
            n_filters,
            activation="tanh",
            kernel_quantizer=block_kernel_quantizer,
            bias_quantizer=block_bias_quantizer,
        )

        # Global exchange + output (Linear(4*n_filters -> n_filters) + Tanh + BN)
        self.gex = GlobalExchange()
        self.out_dense = QDense(
            n_filters,
            activation="tanh",
            kernel_quantizer=block_kernel_quantizer,
            bias_quantizer=block_bias_quantizer,
        )
        self.out_bn = keras.layers.BatchNormalization()

    def call(self, x, training=False):
        # GravNet core
        x = self.gravnet(x, training=training)

        # Post-GravNet
        x = self.bn0(x, training=training)
        x = self.dense0(x)
        x = self.bn1(x, training=training)
        x = self.dense1(x)

        # Global exchange and output
        x = self.gex(x)
        x = self.out_dense(x)
        x = self.out_bn(x, training=training)
        return x


class QGravNetModel(keras.Model):
    def __init__(
        self,
        n_blocks=4,
        n_neighbours=40,
        n_dimensions=4,
        n_filters=96,
        n_propagate=22,
        n_postgn_dense_blocks=4,
        output_dim=4,
        # Quantizers for all dense layers outside the QGravNetLayer
        dense_kernel_quantizer=None,
        dense_bias_quantizer=None,
        # Optional kwargs forwarded into each QGravNetBlock's internal QGravNetLayer
        gravnet_kwargs=None,
        **kwargs,
    ):
        """
        Keras/QKeras port of the Torch hgcal-gravnet GravNetModel for batched (B, V, F) inputs.
        """
        super().__init__(**kwargs)

        if gravnet_kwargs is None:
            gravnet_kwargs = {}

        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        self.output_dim = output_dim

        # Input BN + global exchange + linear to 64 (on 4*input_dim features)
        self.input_bn = keras.layers.BatchNormalization()
        self.input_exchange = GlobalExchange()
        self.input_dense = QDense(
            64,
            activation=None,  # as in Torch
            kernel_quantizer=dense_kernel_quantizer,
            bias_quantizer=dense_bias_quantizer,
        )

        # GravNet blocks:
        # - First block expects 64 input channels (after input_dense)
        # - Later blocks operate on n_filters output channels
        self.blocks = [
            QGravNetBlock(
                n_neighbours=n_neighbours,
                n_dimensions=n_dimensions,
                n_filters=n_filters,
                n_propagate=n_propagate,
                name=f"qgnblock_{i}",
                block_kernel_quantizer=dense_kernel_quantizer,
                block_bias_quantizer=dense_bias_quantizer,
                gravnet_kwargs=gravnet_kwargs,
            )
            for i in range(n_blocks)
        ]

        # Post-GravNet dense layers: repeated (Dense(128, ReLU) + BN)
        self.postgn_dense_layers = []
        for i in range(n_postgn_dense_blocks):
            self.postgn_dense_layers.append(
                QDense(
                    128,
                    activation="relu",
                    kernel_quantizer=dense_kernel_quantizer,
                    bias_quantizer=dense_bias_quantizer,
                )
            )
            self.postgn_dense_layers.append(keras.layers.BatchNormalization())

        # Output head: 128 -> 64 -> 64 -> output_dim
        self.out0 = QDense(
            64,
            activation="relu",
            kernel_quantizer=dense_kernel_quantizer,
            bias_quantizer=dense_bias_quantizer,
        )
        self.out1 = QDense(
            64,
            activation="relu",
            kernel_quantizer=dense_kernel_quantizer,
            bias_quantizer=dense_bias_quantizer,
        )
        self.out2 = QDense(
            output_dim,
            activation=None,  # Torch stops with a linear layer
            kernel_quantizer=dense_kernel_quantizer,
            bias_quantizer=dense_bias_quantizer,
        )

    def call(self, inputs, training=False):
        # inputs: (B, V, F)
        x = self.input_bn(inputs, training=training)
        x = self.input_exchange(x)
        x = self.input_dense(x)

        feat_list = []
        for block in self.blocks:
            x = block(x, training=training)
            feat_list.append(x)

        x = keras.ops.concatenate(feat_list, axis=-1)  # (B, V, n_blocks * n_filters)

        for layer in self.postgn_dense_layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)

        x = self.out0(x)
        x = self.out1(x)
        x = self.out2(x)
        return x
