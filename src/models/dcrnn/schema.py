"""Parameter schema for the DCRNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DCRNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    rnn_units: int = 16
    num_rnn_layers: int = 1
    max_diffusion_step: int = 2
    cl_decay_steps: int = 2000
    use_curriculum_learning: bool = False
