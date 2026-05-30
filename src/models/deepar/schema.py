from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    embedding_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    cov_feat_size: int = 0
    dropout: float = 0.1
