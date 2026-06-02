"""Model registry and dynamic registration helpers."""

from __future__ import annotations

import importlib
from typing import Callable, Type

from pydantic import BaseModel


class ModelRegistry:
    """Registry mapping model names to factory callables and schemas."""

    def __init__(self) -> None:
        self._models: dict[str, tuple[Callable, Type[BaseModel] | None]] = {}

    def register(
        self, name: str, factory: Callable, schema: Type[BaseModel] | None = None
    ) -> None:
        """Register a model factory with an optional parameter schema."""
        self._models[name] = (factory, schema)

    def get(self, name: str) -> tuple[Callable, Type[BaseModel] | None]:
        """Get a model factory and schema by name.

        Raises
        ------
        KeyError
            If the model is not registered.
        """
        if name not in self._models:
            raise KeyError(f"Model '{name}' is not registered")
        return self._models[name]

    def names(self) -> list[str]:
        """Return registered model names."""
        return sorted(self._models.keys())


MODEL_REGISTRY = ModelRegistry()

MODEL_NAME_MAP = {
    "BiMamba": "models.bimamba.registry",
    "WPMixer": "models.wpmixer.registry",
    "DLinear": "models.dlinear.registry",
    "Linear": "models.linear.registry",
    "NLinear": "models.nlinear.registry",
    "RLinear": "models.rlinear.registry",
    "CMoS": "models.cmos.registry",
    "CycleNet": "models.cyclenet.registry",
    "TimeEmb": "models.timeemb.registry",
    "MixLinear": "models.mixlinear.registry",
    "PWS": "models.pws.registry",
    "PaiFilter": "models.paifilter.registry",
    "FITS": "models.fits.registry",
    "SVTime": "models.svtime.registry",
    "SparseTSF": "models.sparsetsf.registry",
    "TexFilter": "models.texfilter.registry",
    "Autoformer": "models.autoformer.registry",
    "FEDformer": "models.fedformer.registry",
    "PatchTST": "models.patchtst.registry",
    "PatchMLP": "models.patchmlp.registry",
    "xPatch": "models.xpatch.registry",
    "Amplifier": "models.amplifier.registry",
    "CrossLinear": "models.crosslinear.registry",
    "TimeBase": "models.timebase.registry",
    "TimeBridge": "models.timebridge.registry",
    "SegRNN": "models.segrnn.registry",
    "TSMixer": "models.tsmixer.registry",
    "LightTS": "models.lightts.registry",
    "SCINet": "models.scinet.registry",
    "TiDE": "models.tide.registry",
    "TimeMixer": "models.timemixer.registry",
    "TimesNet": "models.timesnet.registry",
    "iTransformer": "models.itransformer.registry",
    # Tier 2 / spatiotemporal ports:
    "STNorm": "models.stnorm.registry",
    # Tier 1 / benchmark ports:
    "TimeXer": "models.timexer.registry",
    "TimeFilter": "models.timefilter.registry",
    "MambaSimple": "models.mambasimple.registry",
    "S_Mamba": "models.s_mamba.registry",
    "S4": "models.s4.registry",
    "MSGNet": "models.msgnet.registry",
    "HDMixer": "models.hdmixer.registry",
    "DSFormer": "models.dsformer.registry",
    "UMixer": "models.umixer.registry",
    "TimeKAN": "models.timekan.registry",
    "Fredformer": "models.fredformer.registry",
    "PAttn": "models.pattn.registry",
    "CARD": "models.card.registry",
    "NHiTS": "models.nhits.registry",
    "NBeats": "models.nbeats.registry",
    "DUET": "models.duet.registry",
    "ETSformer": "models.etsformer.registry",
    "NSTransformer": "models.nstransformer.registry",
    "SOFTS": "models.softs.registry",
    "Transformer": "models.transformer.registry",
    "Reformer": "models.reformer.registry",
    "Pyraformer": "models.pyraformer.registry",
    "MultiPatchFormer": "models.multipatchformer.registry",
    "ModernTCN": "models.moderntcn.registry",
    "Crossformer": "models.crossformer.registry",
    "FreTS": "models.frets.registry",
    "FiLM": "models.film.registry",
    "MICN": "models.micn.registry",
    "Koopa": "models.koopa.registry",
    "Informer": "models.informer.registry",
    "MTSMixer": "models.mtsmixer.registry",
    "Pathformer": "models.pathformer.registry",
    "WaveNet": "models.wavenet.registry",
    "DeepAR": "models.deepar.registry",
    "Sumba": "models.sumba.registry",
    "SRSNet": "models.srsnet.registry",
    "DTAF": "models.dtaf.registry",
    "TimePerceiver": "models.timeperceiver.registry",
    "CrossGNN": "models.crossgnn.registry",
    # PyTorch-native classical ML / statistical time-series adapters.
    "RidgeRegressionTS": "models.ridge_regression_ts.registry",
    "LassoRegressionTS": "models.lasso_regression_ts.registry",
    "ElasticNetTS": "models.elastic_net_ts.registry",
    "BayesianRidgeTS": "models.bayesian_ridge_ts.registry",
    "PolynomialRegressionTS": "models.polynomial_regression_ts.registry",
    "KNNForecasterTS": "models.knn_forecaster_ts.registry",
    "SVRForecasterTS": "models.svr_forecaster_ts.registry",
    "GaussianProcessTS": "models.gaussian_process_ts.registry",
    "DecisionTreeTS": "models.decision_tree_ts.registry",
    "RandomForestTS": "models.random_forest_ts.registry",
    "ExtraTreesTS": "models.extra_trees_ts.registry",
    "GradientBoostingTS": "models.gradient_boosting_ts.registry",
    "XGBoostTS": "models.xgboost_ts.registry",
    "LightGBMTS": "models.lightgbm_ts.registry",
    "CatBoostTS": "models.catboost_ts.registry",
    "ARIMATS": "models.arima_ts.registry",
    "AutoRegressiveTS": "models.autoregressive_ts.registry",
    "ExpSmoothingTS": "models.exp_smoothing_ts.registry",
    "KalmanFilterTS": "models.kalman_filter_ts.registry",
    "MLPForecasterTS": "models.mlp_forecaster_ts.registry",
    "RNNForecasterTS": "models.rnn_forecaster_ts.registry",
    "GRUForecasterTS": "models.gru_forecaster_ts.registry",
    "LSTMForecasterTS": "models.lstm_forecaster_ts.registry",
    "TCNForecasterTS": "models.tcn_forecaster_ts.registry",
    # Recent open-source time-series forecasting model adapters (2025/2026 venues).
    "Aurora": "models.aurora.registry",
    "TimeAlign": "models.timealign.registry",
    "GTR": "models.gtr.registry",
    "PhaseFormer": "models.phaseformer.registry",
    "PMDformer": "models.pmdformer.registry",
    "MMPD": "models.mmpd.registry",
    "COSA": "models.cosa.registry",
    "DistDF": "models.distdf.registry",
    "Sonnet": "models.sonnet.registry",
    "APN": "models.apn.registry",
    "TimeCAP": "models.timecap.registry",
    "GOTSF": "models.gotsf.registry",
    "FTP": "models.ftp.registry",
    "OccamVTS": "models.occamvts.registry",
    "HN_MVTS": "models.hn_mvts.registry",
    "SEMPO": "models.sempo.registry",
    "InterPDN": "models.interpdn.registry",
    "TimeO1": "models.timeo1.registry",
    "FeTS": "models.fets.registry",
    "SymTime": "models.symtime.registry",
    "ImplicitForecaster": "models.implicitforecaster.registry",
    "AMRC": "models.amrc.registry",
    "HMformer": "models.hmformer.registry",
    "TiRex": "models.tirex.registry",
    "LatentTSF": "models.latentsf.registry",
    "CoRA": "models.cora.registry",
    "DynamicTMoE": "models.dynamic_tmoe.registry",
    "PULSE": "models.pulse.registry",
    "OLinear": "models.olinear.registry",
    "MAFS": "models.mafs.registry",
    "TSRAG": "models.tsrag.registry",
    "TimeMosaic": "models.timemosaic.registry",
    "Kronos": "models.kronos.registry",
    # Ported PoorOtterBob models.
    # Time-series forecasting:
    "MoFo": "models.mofo.registry",
    "PHAT": "models.phat.registry",
    # Spatiotemporal forecasting:
    "BiST": "models.bist.registry",
    "MAGE": "models.mage.registry",
    "STOP": "models.stop.registry",
    # Air-quality forecasting:
    "CauAir": "models.cauair.registry",
    "AirCade": "models.aircade.registry",
    # Graph spatiotemporal forecasting:
    "GTS": "models.gts.registry",
    # Tier 2 / graph models:
    "STID": "models.stid.registry",
    "GWNet": "models.gwnet.registry",
    "D2STGNN": "models.d2stgnn.registry",
    "DFDGCN": "models.dfdgcn.registry",
    "STGCN": "models.stgcn.registry",
    "AGCRN": "models.agcrn.registry",
    "DCRNN": "models.dcrnn.registry",
    "StemGNN": "models.stemgnn.registry",
    "MTGNN": "models.mtgnn.registry",
    "STGODE": "models.stgode.registry",
    "STAEformer": "models.staeformer.registry",
    "DGCRN": "models.dgcrn.registry",
    "STDN": "models.stdn.registry",
    "STPGNN": "models.stpgnn.registry",
    "MegaCRN": "models.megacrn.registry",
    "HimNet": "models.himnet.registry",
    "STWave": "models.stwave.registry",
    "BigST": "models.bigst.registry",
    # CauAir air-quality models (ported from PoorOtterBob/CauAir).
    # Graph spatiotemporal:
    "ASTGCN": "models.astgcn.registry",
    "GCLSTM": "models.gclstm.registry",
    "DeepAir": "models.deepair.registry",
    "STTN": "models.sttn.registry",
    "GAGNN": "models.gagnn.registry",
    "PM25_GNN": "models.pm25gnn.registry",
    "AirFormer": "models.airformer.registry",
    "DSTAGNN": "models.dstagnn.registry",
    "PCDCNet": "models.pcdcnet.registry",
    "AirPhyNet": "models.airphynet.registry",
    "AirDualODE": "models.airdualode.registry",
    # Non-graph baselines / forecasters:
    "HL": "models.hl.registry",
    "LSTM": "models.lstm.registry",
    "RPMixer": "models.rpmixer.registry",
    "MGSFformer": "models.mgsfformer.registry",
    "CATS": "models.cats.registry",
}

_REGISTERED_MODELS: set[str] = set()


def register_model_by_name(name: str) -> None:
    """Import and register a model using the name map.

    Parameters
    ----------
    name : str
        Model name from the config.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the model name is not mapped.
    ModuleNotFoundError
        If the mapped module cannot be imported.
    AttributeError
        If the module has no register() function.
    """
    if name in _REGISTERED_MODELS:
        return
    module_name = MODEL_NAME_MAP.get(name)
    if module_name is None:
        available = ", ".join(sorted(MODEL_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Model '{name}' is not mapped. Update MODEL_NAME_MAP in "
            f"benchmark.registry.models. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Model registry module not found: {module_name}. "
                "Expected module path in MODEL_NAME_MAP"
            ) from exc
        raise ImportError(
            f"Failed to import '{module_name}' due to missing dependency: {exc}"
        ) from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(
            f"Model registry '{module_name}' must define a register() function"
        )
    register_fn()
    _REGISTERED_MODELS.add(name)
