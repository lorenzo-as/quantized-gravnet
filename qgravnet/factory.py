# pyright: reportMissingImports=false

from typing import Dict, Literal, Optional

from qkeras import QDense
from tensorflow import keras
from tensorflow.keras import layers

from .layers import GlobalExchange
from .core import GravNetCore
from .selectors import REGISTRY as SELECTOR_REGISTRY


class GravNetFactory:
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
        distance_metric: Literal["l1", "l2_squared"] = "l2_squared",
        neighbour_selector: Literal["full", "binned"] = "full",
        gravnet_cfg: Optional[Dict] = None,
        selector_cfg: Optional[Dict] = None,
        dense_layer_dims: Optional[Dict[str, int]] = None,
    ):
        if gravnet_cfg is None:
            gravnet_cfg = {}

        self.n_blocks = n_blocks
        self.n_neighbours = n_neighbours
        self.n_dimensions = n_dimensions
        self.n_filters = n_filters
        self.n_propagate = n_propagate
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        self.output_dim = output_dim
        self.output_head = output_head
        self.distance_metric = distance_metric
        self.neighbour_selector = SELECTOR_REGISTRY[neighbour_selector](**(selector_cfg or {}))
        self.gravnet_cfg = gravnet_cfg

        self.dense_layer_dims = {
            "input_dense": 64,
            "post_gn": 128,
            "postgn_block": 128,
            "out0": 64,
            "out1": 64,
        }
        if dense_layer_dims:
            self.dense_layer_dims.update(dense_layer_dims)

    def _make_dense(
        self,
        units: int,
        activation: Optional[str] = None,
        kernel_initializer: str = "glorot_uniform",
        name: Optional[str] = None,
        kernel_quantizer=None,
        bias_quantizer=None,
    ):
        """Create a dense layer. Subclasses override to use QDense with quantizers."""
        return layers.Dense(
            units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            name=name,
        )

    def create_keras_model(self, n_vertices: int, n_features: int) -> keras.Model:
        inputs = keras.Input(shape=(n_vertices, n_features), name="gravnet_input")

        # Input BN + global exchange + linear to 64
        x = GlobalExchange(name="input_gex")(inputs)
        x = self._make_dense(
            self.dense_layer_dims["input_dense"],
            activation=None,
            name="input_dense",
        )(x)

        for ib in range(self.n_blocks):
            block_prefix = f"qgnblock_{ib}"
            gkw = self.gravnet_cfg.copy()

            # Input feature transform: F_in → F_prop
            input_feature_transform = self._make_dense(
                self.n_propagate,
                kernel_initializer=gkw.get("other_kernel_initializer", "glorot_uniform"),
                kernel_quantizer=gkw.get("feature_kernel_quantizer", None),
                bias_quantizer=gkw.get("feature_bias_quantizer", None),
                activation=gkw.get("feature_activation", None),
                name=f"{block_prefix}_input_feature_transform",
            )

            # Input spatial transform: F_in → S (latent coordinate space)
            if not gkw.get("fix_coordinate_space", False):
                input_spatial_transform = self._make_dense(
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
                raise ValueError("fix_coordinate_space=False required")

            # Output feature transform: concat([x, neigh]) → n_filters
            output_feature_transform = self._make_dense(
                self.n_filters,
                kernel_initializer=gkw.get("other_kernel_initializer", "glorot_uniform"),
                kernel_quantizer=gkw.get("output_kernel_quantizer", None),
                bias_quantizer=gkw.get("output_bias_quantizer", None),
                activation=gkw.get("output_activation", "tanh"),
                name=f"{block_prefix}_output_feature_transform",
            )

            core = GravNetCore(
                self.n_neighbours,
                distance_metric=self.distance_metric,
                selector=self.neighbour_selector,
                name=f"{block_prefix}_core",
            )

            fprop = input_feature_transform(x)
            if 0.0 < gkw.get("feature_dropout", -1.0) < 1.0:
                fprop = layers.Dropout(
                    gkw.get("feature_dropout"), name=f"{block_prefix}_dropout"
                )(fprop)

            coords = input_spatial_transform(x)
            neigh = core([coords, fprop])
            merged = layers.Concatenate(name=f"{block_prefix}_merge")([x, neigh])

            out = output_feature_transform(merged)

            out = self._make_dense(
                self.dense_layer_dims["post_gn"],
                activation=gkw.get("post_gn_activation", "tanh"),
                kernel_initializer=gkw.get("other_kernel_initializer", "glorot_uniform"),
                name=f"{block_prefix}_dense0",
            )(out)
            out = self._make_dense(
                self.n_filters,
                activation=gkw.get("post_gn_out_activation", "tanh"),
                kernel_initializer=gkw.get("other_kernel_initializer", "glorot_uniform"),
                name=f"{block_prefix}_dense1",
            )(out)

            out = GlobalExchange(name=f"{block_prefix}_gex")(out)
            out = self._make_dense(
                self.n_filters,
                activation=gkw.get("post_gn_gex_activation", "tanh"),
                kernel_initializer=gkw.get("other_kernel_initializer", "glorot_uniform"),
                name=f"{block_prefix}_out_dense",
            )(out)

            x = out

        # Post-GravNet dense layers
        for i in range(self.n_postgn_dense_blocks):
            x = self._make_dense(
                self.dense_layer_dims["postgn_block"],
                activation=gkw.get("post_gn_relu", "relu"),
                name=f"postgn_dense_{i}",
            )(x)

        x = self._make_dense(
            self.dense_layer_dims["out0"],
            activation=gkw.get("post_gn_relu", "relu"),
            name="out0",
        )(x)
        x = self._make_dense(
            self.dense_layer_dims["out1"],
            activation=gkw.get("post_gn_relu", "relu"),
            name="out1",
        )(x)

        if self.output_head == "dual":
            x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
            energies = self._make_dense(
                1,
                activation=None,
                kernel_quantizer=gkw.get("regression_kernel_quantizer", None),
                bias_quantizer=gkw.get("regression_bias_quantizer", None),
                name="regression",
            )(x)
            classes = self._make_dense(
                1,
                activation=gkw.get("classification_activation", "sigmoid"),
                name="classification",
            )(x)

            return keras.Model(
                inputs=inputs, outputs=[energies, classes], name="qgravnet_model_dual"
            )

        elif self.output_head == "oc":
            outputs = self._make_dense(
                self.output_dim,
                activation=None,
                name="out2",
            )(x)

            return keras.Model(inputs=inputs, outputs=outputs, name="qgravnet_model_oc")


class QGravNetFactory(GravNetFactory):
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
        distance_metric: Literal["l1", "l2_squared"] = "l2_squared",
        neighbour_selector: Literal["full", "binned"] = "full",
        dense_kernel_quantizer=None,
        dense_bias_quantizer=None,
        gravnet_cfg: Optional[Dict] = None,
        selector_cfg: Optional[Dict] = None,
        dense_layer_dims: Optional[Dict[str, int]] = None,
    ):
        super().__init__(
            n_blocks=n_blocks,
            n_neighbours=n_neighbours,
            n_dimensions=n_dimensions,
            n_filters=n_filters,
            n_propagate=n_propagate,
            n_postgn_dense_blocks=n_postgn_dense_blocks,
            output_dim=output_dim,
            output_head=output_head,
            distance_metric=distance_metric,
            neighbour_selector=neighbour_selector,
            gravnet_cfg=gravnet_cfg,
            selector_cfg=selector_cfg,
            dense_layer_dims=dense_layer_dims,
        )
        self.dense_kernel_quantizer = dense_kernel_quantizer
        self.dense_bias_quantizer = dense_bias_quantizer

    def _make_dense(
        self,
        units: int,
        activation: Optional[str] = None,
        kernel_initializer: str = "glorot_uniform",
        name: Optional[str] = None,
        kernel_quantizer=None,
        bias_quantizer=None,
    ):
        return QDense(
            units,
            activation=activation,
            kernel_initializer=kernel_initializer,
            kernel_quantizer=kernel_quantizer or self.dense_kernel_quantizer,
            bias_quantizer=bias_quantizer or self.dense_bias_quantizer,
            name=name,
        )