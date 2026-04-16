# pyright: reportMissingImports=false


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


@register("full")
@keras.saving.register_keras_serializable(package="qgravnet")
class FullSelector(NeighbourSelector):
    """No restriction, use full pairwise distance matrix for neighbour search."""

    def restrict(self, dist, coords):
        return dist
