"""Parameter schema for the STID model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STID parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset, so they are not
    declared here. Defaults are modest for fast smoke runs.
    """

    enc_in: int
    input_dim: int = 3
    embed_dim: int = 32
    num_layers: int = 1
    num_time_in_day: int = 24
    num_day_in_week: int = 7
    if_time_in_day: bool = True
    if_day_in_week: bool = True
