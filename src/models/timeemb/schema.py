from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 512
    use_revin: bool = True
    use_hour_index: bool = True
    use_day_index: bool = False
    scale: float = 0.02
    hour_length: int = 24
    day_length: int = 7
    rec_lambda: float = 0.0
    auxi_lambda: float = 1.0
    auxi_loss: str = "MAE"
    auxi_mode: str = "fft"
    auxi_type: str = "complex"
    module_first: bool = True
    leg_degree: int = 2
    add_noise: bool = False
