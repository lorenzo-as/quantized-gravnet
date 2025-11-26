# quantized-gravnet

A QKerasV3-based implementation of the **GravNet** architecture.

This package provides:
- `GravNetCore` - the core neighbour-aggregation logic
- `GlobalExchange` - feature broadcasting (mean/min/max)
- `QGravNetLayer` - a quantized GravNet layer
- `QGravNetBlock` and `QGravNetModel` - modular, multi-block GravNet using Keras subclassing
- `QGravNetFactory` - factory class to build quantized GravNet models using the Keras Functional API

## Installation

This package depends on QKeras v3 (available on TestPyPI for now):

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple qkeras-v3==1.0.3
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

```
from qgravnet import QGravNetFactory
model = QGravNetFactory(n_blocks=4, n_neighbours=40).create_keras_model(n_vertices=128, n_features=16)

model.summary()
```
