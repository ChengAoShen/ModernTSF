from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Parameters for the CRIB model.

    Note: ``patch_len`` must divide ``task.seq_len``; ``model_dim`` must be
    divisible by ``heads_num``.
    """

    enc_in: int
    patch_len: int = 8
    model_dim: int = 32
    heads_num: int = 4
    enc_num: int = 3
    dropout: float = 0.1
    activation: str = "relu"
    # Consistency (MSE) and IB (KL) regularizer weights (upstream 1.0 / 1e-6).
    consis_weight: float = 1.0
    kl_weight: float = 1e-6
