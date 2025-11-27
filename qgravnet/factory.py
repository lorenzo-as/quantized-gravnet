# pyright: reportMissingImports=false

from typing import Dict, Literal, Optional

import keras
from qkeras import QDense

from .layers import GlobalExchange, GravNetCore


class QGravNetFactory:
    def __init__(
        self,
        n_blocks: int = 4,
        n_neighbours: int = 40,
        n_dimensions: int = 4,
        n_filters: int = 96,
        n_propagate: int = 22,
        n_postgn_dense_blocks: int = 4,
        output_dim: int = 4,
        output_head: Literal["dual", "oc"] = "dual",
        dense_kernel_quantizer=None,
        dense_bias_quantizer=None,
        gravnet_kwargs: Optional[Dict] = None,
    ):
        if gravnet_kwargs is None:
            gravnet_kwargs = {}

        self.n_blocks = n_blocks
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        self.output_dim = output_dim
        self.output_head = output_head

        self.dense_kernel_quantizer = dense_kernel_quantizer
        self.dense_bias_quantizer = dense_bias_quantizer
        self.gravnet_kwargs = gravnet_kwargs

    def create_keras_model(self, n_vertices: int, n_features: int) -> keras.Model:
        inputs = keras.Input(shape=(n_vertices, n_features), name="input")

        # Input BN + global exchange + linear to 64 (on 4*input_dim features)
        x = keras.layers.BatchNormalization(name="input_bn")(inputs)
        x = GlobalExchange(name="input_gex")(x)
        x = QDense(
            64,
            activation=None,
            kernel_quantizer=self.dense_kernel_quantizer,
            bias_quantizer=self.dense_bias_quantizer,
            name="input_dense",
        )(x)

        feat_list = []
        for ib in range(self.n_blocks):
            block_prefix = f"qgnblock_{ib}"
            gkw = self.gravnet_kwargs.copy()

            # Input feature transform: F_in -> F_prop
            input_feature_transform = QDense(
                self.n_propagate,
                kernel_initializer=gkw.get(
                    "other_kernel_initializer", "glorot_uniform"
                ),
                kernel_quantizer=gkw.get("feature_kernel_quantizer", None),
                bias_quantizer=gkw.get("feature_bias_quantizer", None),
                activation=gkw.get("feature_activation", None),
                name=f"{block_prefix}_input_feature_transform",
            )

            # Input spatial transform: F_in -> S
            if not gkw.get("fix_coordinate_space", False):
                input_spatial_transform = QDense(
                    self.n_dimensions,
                    kernel_initializer=gkw.get(
                        "coordinate_kernel_initializer", keras.initializers.Orthogonal()
                    ),
                    kernel_quantizer=gkw.get("coordinate_kernel_quantizer", None),
                    bias_quantizer=gkw.get("coordinate_bias_quantizer", None),
                    activation=gkw.get("coordinate_activation", None),
                    name=f"{block_prefix}_input_spatial_transform",
                )
            else:
                raise ValueError(
                    "QGravNetFactory requires fix_coordinate_space=False for now"
                )

            # Output transform concat([x, neigh]) -> n_filters
            output_feature_transform = QDense(
                self.n_filters,
                kernel_initializer=gkw.get(
                    "other_kernel_initializer", "glorot_uniform"
                ),
                kernel_quantizer=gkw.get("output_kernel_quantizer", None),
                bias_quantizer=gkw.get("output_bias_quantizer", None),
                activation=gkw.get("output_activation", "tanh"),
                name=f"{block_prefix}_output_feature_transform",
            )

            core = GravNetCore(self.n_neighbours, name=f"{block_prefix}_core")

            fprop = input_feature_transform(x)
            if 0.0 < gkw.get("feature_dropout", -1.0) < 1.0:
                fprop = keras.layers.Dropout(
                    gkw.get("feature_dropout"), name=f"{block_prefix}_dropout"
                )(fprop)

            coords = input_spatial_transform(x)
            neigh = core(coords, fprop)
            merged = keras.layers.Concatenate(name=f"{block_prefix}_merge")([x, neigh])

            out = output_feature_transform(merged)

            # Post-GravNet: BN -> Dense(128,tanh) -> BN -> Dense(n_filters,tanh)
            out = keras.layers.BatchNormalization(name=f"{block_prefix}_bn0")(out)
            out = QDense(
                128,
                activation="tanh",
                kernel_quantizer=self.dense_kernel_quantizer,
                bias_quantizer=self.dense_bias_quantizer,
                name=f"{block_prefix}_dense0",
            )(out)
            out = keras.layers.BatchNormalization(name=f"{block_prefix}_bn1")(out)
            out = QDense(
                self.n_filters,
                activation="tanh",
                kernel_quantizer=self.dense_kernel_quantizer,
                bias_quantizer=self.dense_bias_quantizer,
                name=f"{block_prefix}_dense1",
            )(out)

            # Global exchange + output (Linear(4*n_filters -> n_filters) + Tanh + BN)
            out = GlobalExchange(name=f"{block_prefix}_gex")(out)
            out = QDense(
                self.n_filters,
                activation="tanh",
                kernel_quantizer=self.dense_kernel_quantizer,
                bias_quantizer=self.dense_bias_quantizer,
                name=f"{block_prefix}_out_dense",
            )(out)
            out = keras.layers.BatchNormalization(name=f"{block_prefix}_out_bn")(out)

            feat_list.append(out)
            x = out

        # Concatenate features from all blocks - do this iteratively for hls4ml compatibility
        for i, feat in enumerate(feat_list, start=1):
            if i == 1:
                x = feat
            else:
                x = keras.layers.Concatenate(name=f"final_concat_{i}")([x, feat])

        # Post-GravNet dense layers: repeated (Dense(128, ReLU) + BN)
        for i in range(self.n_postgn_dense_blocks):
            x = QDense(
                128,
                activation="relu",
                kernel_quantizer=self.dense_kernel_quantizer,
                bias_quantizer=self.dense_bias_quantizer,
                name=f"postgn_dense_{i}",
            )(x)
            x = keras.layers.BatchNormalization(name=f"postgn_bn_{i}")(x)

        # Output head: 128 -> 64 -> 64 -> output_dim
        x = QDense(
            64,
            activation="relu",
            kernel_quantizer=self.dense_kernel_quantizer,
            bias_quantizer=self.dense_bias_quantizer,
            name="out0",
        )(x)
        x = QDense(
            64,
            activation="relu",
            kernel_quantizer=self.dense_kernel_quantizer,
            bias_quantizer=self.dense_bias_quantizer,
            name="out1",
        )(x)

        if self.output_head == "dual":
            x = keras.layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
            energies = QDense(
                1,
                activation=None,
                kernel_quantizer=self.dense_kernel_quantizer,
                bias_quantizer=self.dense_bias_quantizer,
                name="regression",
            )(x)
            classes = QDense(
                1,
                activation="sigmoid",
                kernel_quantizer=self.dense_kernel_quantizer,
                bias_quantizer=self.dense_bias_quantizer,
                name="classification",
            )(x)

            return keras.Model(
                inputs=inputs, outputs=[energies, classes], name="qgravnet_model_dual"
            )

        elif self.output_head == "oc":
            outputs = QDense(
                self.output_dim,
                activation=None,
                kernel_quantizer=self.dense_kernel_quantizer,
                bias_quantizer=self.dense_bias_quantizer,
                name="out2",
            )(x)

            return keras.Model(inputs=inputs, outputs=outputs, name="qgravnet_model_oc")
