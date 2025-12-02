# pyright: reportMissingImports=false

from tensorflow.keras import layers


def pairwise_concatenate(tensors, axis=-1, name_prefix="pairwise_concat"):
    """
    Iteratively concatenates a list of tensors using binary Keras Concatenate
    layers (used for hls4ml compatibility).

    Parameters:
        tensors (list): List of tensors to concatenate.
        axis (int): Concatenation axis.
        name_prefix (str): Prefix for generated layer names.

    Returns:
        keras.Tensor: The pairwise-concatenated output tensor.
    """
    if not tensors:
        raise ValueError("Expected at least one tensor, got an empty list.")

    if len(tensors) == 1:
        return tensors[0]

    out = tensors[0]
    for i, t in enumerate(tensors[1:], start=1):
        out = layers.Concatenate(axis=axis, name=f"{name_prefix}_{i}")([out, t])
    return out
