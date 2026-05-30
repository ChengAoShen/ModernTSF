from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    ffn_ratio: int = 1
    num_blocks: list[int] = [1]
    large_size: list[int] = [13]
    small_size: list[int] = [5]
    dims: list[int] = [32]
    dw_dims: list[int] = [32]
    patch_size: int = 16
    patch_stride: int = 16
    stem_ratio: int = 6
    downsample_ratio: int = 2
    small_kernel_merged: bool = False
    dropout: float = 0.1
    head_dropout: float = 0.1
    use_multi_scale: bool = True
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    individual: bool = False
    decomposition: bool = False
    kernel_size: int = 25
