from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    e_layers: int = 2
    anti_ood: bool = True
    tk: int = 10
    scale_number: int = 4
    use_tgcn: bool = True
    use_ngcn: bool = True
    individual: bool = False
    dropout: float = 0.1
    tvechidden: int = 8
    nvechidden: int = 8
    hidden: int = 16
