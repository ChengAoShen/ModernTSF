from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    alias: str | None = None
    # TSEval leaderboard track for runs on this dataset. Defaults to the task
    # mode (time_series / spatiotemporal / covariate) when unset; set it to
    # "realtime" for periodically-refreshed live datasets (e.g. stock_hs300).
    track: str | None = None
    root_path: str = "./data/"
    data_path: str
    params: dict
