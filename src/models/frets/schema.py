from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    embed_size: int = 128
    hidden_size: int = 256
    channel_independence: bool = False
