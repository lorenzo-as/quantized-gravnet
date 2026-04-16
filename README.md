# quantized-gravnet

A Tensorflow + QKeras-based implementation of the **GravNet** architecture.

This package provides:
- `GravNetCore` - the core neighbour-aggregation logic
- `GlobalExchange` - feature broadcasting (mean/min/max)
- `GravNetLayer` and `QGravNetLayer` - a (quantized) GravNet layer
- `QGravNetBlock` and `QGravNetModel` - modular, multi-block GravNet using Keras subclassing
- `GravNetFactory` and `QGravNetFactory` - factory class to build (quantized) GravNet models using the Keras Functional API
- `NeighbourSelector` - extendable neighbour selection strategies for `GravNetCore` 

## Installation

This package is pinned for reproducibility with:

- **TensorFlow 2.14.0**
- **QKeras 0.9.0**

```bash
pip install "quantized-gravnet @ git+https://github.com/lorenzo-as/quantized-gravnet.git@v0.1.0"
```

Development install:

```bash
git clone https://github.com/lorenzo-as/quantized-gravnet.git
cd quantized-gravnet
pip install -e ".[dev]"
```

## API notes

### `GravNetCore.call()` signature

`GravNetCore` expects a **list/tuple of tensors**:

- `coords`: `(B, V, S)`
- `feats`: `(B, V, F_prop)`

```python
aggregated = core([coords, feats])
```

### `QGravNetLayer.call()` / `GravNetLayer.call()` signature

Layer call takes a **single feature tensor**:

- `x`: `(B, V, F_in)`

```python
y = layer(x)
```

## Reference to Original GravNet Paper

```
@article{Qasim:2019otl,
title = "{Learning representations of irregular particle-detector geometry with distance-weighted graph networks}",
author = "Qasim, Shah Rukh and Kieseler, Jan and Iiyama, Yutaro and Pierini, Maurizio",
journal = "Eur. Phys. J. C79 (7) 608",
year = "2019",
eprint = "1902.07987",
doi = "10.1140/epjc/s10052-019-7113-9"
}
```

**GravNet Keras reference implementation:**
https://github.com/jkiesele/caloGraphNN/blob/master/keras_models.py


## Example

### Non-quantized model

```python
from qgravnet.factory import GravNetFactory
model = GravNetFactory(n_blocks=4, n_neighbours=40).create_keras_model(n_vertices=128, n_features=16)

model.summary()
```

### Quantized model

```python
from qgravnet import QGravNetFactory
model = QGravNetFactory(n_blocks=4, n_neighbours=40).create_keras_model(n_vertices=128, n_features=16)

model.summary()
```
