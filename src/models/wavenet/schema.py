from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    use_norm: bool = True
