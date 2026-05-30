from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    layer_nums: int = 2
    k: int = 2
    num_experts: int = 4
    # Flat list of length layer_nums * num_experts, reshaped to per-layer
    # patch sizes. Each value must divide seq_len evenly.
    patch_size_list: list[int] = [16, 12, 8, 6, 16, 12, 8, 6]
    d_model: int = 16
    d_ff: int = 64
    residual_connection: int = 1
    revin: bool = True
    batch_norm: bool = False
