from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    input_dim: int = 1
    output_dim: int = 1
    residual_channels: int = 16
    conv_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    dimension: int = 16
    M: int = 4
    LowRank: int = 8
    D: int = 16
    gcn_depth: int = 2
    sumba_layers: int = 2
    layers: int = 2
    dilation_exponential: int = 1
    kernel_set: list[int] = [2, 3, 6, 7]
    propalpha: float = 0.05
    dropout: float = 0.3
    layer_norm_affline: bool = True
    mark_dim: int = 6
