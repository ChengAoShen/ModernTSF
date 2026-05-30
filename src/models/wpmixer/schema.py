from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    dropout: float = 0.1
    tfactor: int = 5
    dfactor: int = 5
    wavelet: str = "db2"
    level: int = 1
    patch_len: int = 16
    stride: int = 8
    no_decomposition: bool = False
