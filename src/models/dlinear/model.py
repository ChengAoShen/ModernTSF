"""Public DLinear model wrapper over the shared decomposition backbone."""

from components.dlinear import DLinearBackbone


class Model(DLinearBackbone):
    """Named DLinear catalog entry; behavior is defined by ``DLinearBackbone``."""
