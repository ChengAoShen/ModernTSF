"""Parameter schema for the MegaCRN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MegaCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are *injected* by the runner from the dataset
    (see ``src/benchmark/runner/run_one.py``) and need not be declared in TOML.
    """

    enc_in: int  # number of spatial nodes N (required)
    input_dim: int = 3
    rnn_units: int = 32
    num_layers: int = 1
    cheb_k: int = 3
    mem_num: int = 8
    mem_dim: int = 16
    use_curriculum_learning: bool = True
