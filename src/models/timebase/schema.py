from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period_len: int = 24
    basis_num: int = 6
    individual: bool = False
    use_orthogonal: bool = True
    use_period_norm: bool = True
