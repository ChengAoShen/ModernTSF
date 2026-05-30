"""Parameter schema for the STGODE model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGODE parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
