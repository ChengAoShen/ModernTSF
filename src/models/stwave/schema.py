"""Parameter schema for the STWave model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STWave parameters supplied via ``model.params``.

    ``enc_in`` (the number of spatial nodes ``N``) is required. ``num_nodes``
    and ``adj_mx`` are injected by the runner from the dataset and are NOT
    declared in the TOML. ``hidden_size`` doubles as the number of Laplacian
    eigenvectors used for the spatial positional encoding, so it is clamped to
    ``N`` at construction.
    """

    enc_in: int
    input_dim: int = 3
    hidden_size: int = 16
    layers: int = 1
    log_samples: int = 1
    time_in_day_size: int = 24
    day_in_week_size: int = 7
    wave_type: str = "sym2"
    wave_levels: int = 1
