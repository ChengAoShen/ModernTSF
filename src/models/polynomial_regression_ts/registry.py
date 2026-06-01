"""Model registration for PolynomialRegressionTS."""

from benchmark.registry import MODEL_REGISTRY
from models.polynomial_regression_ts.model import Model
from models.polynomial_regression_ts.schema import ModelParameterConfig


def register() -> None:
    """Register PolynomialRegressionTS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PolynomialRegressionTS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            dropout=params.get("dropout", 0.1),
            num_layers=params.get("num_layers", 1),
            num_estimators=params.get("num_estimators", 16),
            tree_depth=params.get("tree_depth", 3),
            num_prototypes=params.get("num_prototypes", 32),
            kernel_gamma=params.get("kernel_gamma", 0.1),
            l1_penalty=params.get("l1_penalty", 0.0),
            l2_penalty=params.get("l2_penalty", 0.0),
            use_revin=bool(params.get("use_revin", True)),
        ),
        ModelParameterConfig,
    )
