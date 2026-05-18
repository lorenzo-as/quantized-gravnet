# pyright: reportMissingImports=false

import tensorflow as tf
from tensorflow import keras

REGISTRY = {}


def register(name):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls

    return decorator


class NeighbourSelector:
    """Interface for restricting neighbour search in GravNetCore."""

    def restrict(self, dist, coords):
        """Restrict the distance matrix for neighbour search.

        Args:
            dist: (B, V, V) pairwise distance matrix
            coords: (B, V, S) coordinates used for neighbour selection

        Returns:
            Modified dist with the same shape where non-neighbour entries are set to a large value.
        """
        raise NotImplementedError

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@register("full")
@keras.utils.register_keras_serializable(package="qgravnet")
class FullSelector(NeighbourSelector):
    """No restriction, use full pairwise distance matrix for neighbour search."""

    def restrict(self, dist, coords):
        return dist


@register("binned")
@keras.utils.register_keras_serializable(package="qgravnet")
class BinnedSelector(NeighbourSelector):
    """
    Restrict neighbour search to vertices in the same or adjacent bins
    in the learned S-dimensional coordinate space.

    Args:
        bins_per_axis (int):
            Number of bins per coordinate axis.
            Total number of bins = bins_per_axis ** S.
        window (int):
            Bin radius. A value of 1 means same bin ±1 in each dimension.
            Total bins considered per vertex = (2*window + 1) ** S.
        clip_min (float):
            Minimum coordinate value for binning. Coordinates below this will be clipped to this value.
        clip_max (float):
            Maximum coordinate value for binning. Coordinates above this will be clipped to this value.
    """

    def __init__(
        self,
        bins_per_axis: int,
        window: int = 1,
        clip_min: float = -1.0,
        clip_max: float = 1.0,
    ):
        self.bins_per_axis = bins_per_axis
        self.window = window
        self.clip_min = clip_min
        self.clip_max = clip_max

    def get_config(self):
        return {
            "bins_per_axis": self.bins_per_axis,
            "window": self.window,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
        }

    def compute_bins(self, coords):
        # coords: (B, V, S)
        coords = tf.clip_by_value(coords, self.clip_min, self.clip_max)

        scaled = (coords - self.clip_min) / (self.clip_max - self.clip_min)
        scaled = scaled * self.bins_per_axis

        bins = tf.cast(tf.floor(scaled), tf.int32)
        bins = tf.clip_by_value(bins, 0, self.bins_per_axis - 1)
        return bins

    def restrict(self, dist, coords):
        bins = self.compute_bins(coords)  # (B, V, S)

        # compute bin mask: keep pairs of vertices whose bins are within window in all dimensions
        bins_i = bins[:, :, None, :]  # (B, V, 1, S)
        bins_j = bins[:, None, :, :]  # (B, 1, V, S)

        diff = tf.abs(bins_i - bins_j)  # (B, V, V, S)
        mask = tf.reduce_all(diff <= self.window, axis=-1)  # (B, V, V)

        # exclude bins outside window
        large = tf.constant(1e9, dtype=dist.dtype)
        dist_masked = tf.where(mask, dist, large)
        return dist_masked

    def clipping_statistics(self, coords):
        below = coords < self.clip_min
        above = coords > self.clip_max

        total = tf.size(coords, out_type=tf.float32)
        n_below = tf.reduce_sum(tf.cast(below, tf.float32))
        n_above = tf.reduce_sum(tf.cast(above, tf.float32))

        return {
            "n_below": n_below,
            "n_above": n_above,
            "fraction_clipped": tf.math.divide_no_nan(n_below + n_above, total),
        }
