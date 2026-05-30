from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    n_heads: int = 2
    d_ff: int = 256
    patch_len: int = 16
    dropout: float = 0.1
    num_latents: int = 8
    latent_dim: int = 128
    latent_d_ff: int = 256
    num_latent_blocks: int = 1
    use_latent: bool = True
    query_share: bool = True
