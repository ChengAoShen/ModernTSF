from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Parameters for the faithful TimeAlign model.

    Note: ``patch_num`` must divide both ``task.seq_len`` and ``task.pred_len``.
    """

    enc_in: int
    patch_num: int = 4
    d_model: int = 32
    d_ff: int = 32
    e_layers: int = 2
    dropout: float = 0.1
    pos: bool = True
    layer_norm: bool = True
    loc: bool = True
    glo: bool = True
    local_margin: float = 0.0
    global_margin: float = 0.0
    w_recon: float = 1.0
    w_align: float = 0.1
