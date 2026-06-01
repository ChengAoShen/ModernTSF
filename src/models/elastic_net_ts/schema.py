from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    num_layers: int = 1
    num_estimators: int = 16
    tree_depth: int = 3
    num_prototypes: int = 32
    kernel_gamma: float = 0.1
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    use_revin: bool = True
