"""Flat, lazy model catalog and model specification contracts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Literal, Type

from pydantic import BaseModel


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    model_class: type
    factory: Callable
    params_schema: Type[BaseModel]
    config_path: str = ""
    model_card: str = ""
    smoke_config: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)
    components: tuple[str, ...] = ()
    contract_task: dict[str, int | str] = field(default_factory=dict)
    contract_seeds: tuple[int, ...] = (0,)

    @property
    def output_type(self) -> Literal["point", "quantile", "distribution"]:
        """Declared public output kind, derived from orthogonal capabilities."""
        if "quantile-output" in self.capabilities:
            return "quantile"
        if "distribution-output" in self.capabilities:
            return "distribution"
        return "point"

    def validate_params(self, params: dict) -> dict:
        unknown = sorted(set(params) - set(self.params_schema.model_fields))
        if unknown:
            raise ValueError(
                f"Unknown parameters for {self.name}: {', '.join(unknown)}"
            )
        return self.params_schema.model_validate(params).model_dump()

    def build(self, cfg, params: dict):
        return self.factory(cfg, self.validate_params(params))


class ModelCatalog:
    def __init__(self, refs: dict[str, str]) -> None:
        self._refs = dict(refs)
        self._loaded: dict[str, ModelSpec] = {}

    def names(self) -> list[str]:
        return sorted(self._refs)

    def refs(self) -> dict[str, str]:
        return dict(self._refs)

    def get(self, name: str) -> ModelSpec:
        if name in self._loaded:
            return self._loaded[name]
        module_path = self._refs.get(name)
        if module_path is None:
            available = ", ".join(self.names())
            raise KeyError(f"Unknown model {name!r}. Available: {available}")
        module = importlib.import_module(module_path)
        spec = getattr(module, "SPEC", None)
        if not isinstance(spec, ModelSpec):
            raise TypeError(f"{module_path} must expose a ModelSpec named SPEC")
        if spec.name != name:
            raise ValueError(
                f"catalog key {name!r} disagrees with {module_path}.SPEC.name={spec.name!r}"
            )
        self._loaded[name] = spec
        return spec


MODEL_CATALOG = ModelCatalog({
    'BiMamba': 'models.bimamba.spec',
    'WPMixer': 'models.wpmixer.spec',
    'DLinear': 'models.dlinear.spec',
    'Linear': 'models.linear.spec',
    'NLinear': 'models.nlinear.spec',
    'RLinear': 'models.rlinear.spec',
    'CMoS': 'models.cmos.spec',
    'CycleNet': 'models.cyclenet.spec',
    'TimeEmb': 'models.timeemb.spec',
    'MixLinear': 'models.mixlinear.spec',
    'PWS': 'models.pws.spec',
    'PaiFilter': 'models.paifilter.spec',
    'FITS': 'models.fits.spec',
    'SVTime': 'models.svtime.spec',
    'SparseTSF': 'models.sparsetsf.spec',
    'TexFilter': 'models.texfilter.spec',
    'Autoformer': 'models.autoformer.spec',
    'FEDformer': 'models.fedformer.spec',
    'PatchTST': 'models.patchtst.spec',
    'PatchMLP': 'models.patchmlp.spec',
    'xPatch': 'models.xpatch.spec',
    'Amplifier': 'models.amplifier.spec',
    'CrossLinear': 'models.crosslinear.spec',
    'TimeBase': 'models.timebase.spec',
    'TimeBridge': 'models.timebridge.spec',
    'SegRNN': 'models.segrnn.spec',
    'TSMixer': 'models.tsmixer.spec',
    'LightTS': 'models.lightts.spec',
    'SCINet': 'models.scinet.spec',
    'TiDE': 'models.tide.spec',
    'TimeMixer': 'models.timemixer.spec',
    'TimesNet': 'models.timesnet.spec',
    'iTransformer': 'models.itransformer.spec',
    'STNorm': 'models.stnorm.spec',
    'TimeXer': 'models.timexer.spec',
    'TimeFilter': 'models.timefilter.spec',
    'MambaSimple': 'models.mambasimple.spec',
    'S_Mamba': 'models.s_mamba.spec',
    'S4': 'models.s4.spec',
    'MSGNet': 'models.msgnet.spec',
    'HDMixer': 'models.hdmixer.spec',
    'DSFormer': 'models.dsformer.spec',
    'UMixer': 'models.umixer.spec',
    'TimeKAN': 'models.timekan.spec',
    'Fredformer': 'models.fredformer.spec',
    'PAttn': 'models.pattn.spec',
    'CARD': 'models.card.spec',
    'NHiTS': 'models.nhits.spec',
    'NBeats': 'models.nbeats.spec',
    'DUET': 'models.duet.spec',
    'ETSformer': 'models.etsformer.spec',
    'NSTransformer': 'models.nstransformer.spec',
    'SOFTS': 'models.softs.spec',
    'Transformer': 'models.transformer.spec',
    'Reformer': 'models.reformer.spec',
    'Pyraformer': 'models.pyraformer.spec',
    'MultiPatchFormer': 'models.multipatchformer.spec',
    'ModernTCN': 'models.moderntcn.spec',
    'Crossformer': 'models.crossformer.spec',
    'FreTS': 'models.frets.spec',
    'FiLM': 'models.film.spec',
    'MICN': 'models.micn.spec',
    'Koopa': 'models.koopa.spec',
    'Informer': 'models.informer.spec',
    'MTSMixer': 'models.mtsmixer.spec',
    'Pathformer': 'models.pathformer.spec',
    'WaveNet': 'models.wavenet.spec',
    'DeepAR': 'models.deepar.spec',
    'Sumba': 'models.sumba.spec',
    'SRSNet': 'models.srsnet.spec',
    'DTAF': 'models.dtaf.spec',
    'TimePerceiver': 'models.timeperceiver.spec',
    'CrossGNN': 'models.crossgnn.spec',
    'RidgeRegressionTS': 'models.ridge_regression_ts.spec',
    'LassoRegressionTS': 'models.lasso_regression_ts.spec',
    'ElasticNetTS': 'models.elastic_net_ts.spec',
    'BayesianRidgeTS': 'models.bayesian_ridge_ts.spec',
    'PolynomialRegressionTS': 'models.polynomial_regression_ts.spec',
    'KNNForecasterTS': 'models.knn_forecaster_ts.spec',
    'SVRForecasterTS': 'models.svr_forecaster_ts.spec',
    'GaussianProcessTS': 'models.gaussian_process_ts.spec',
    'DecisionTreeTS': 'models.decision_tree_ts.spec',
    'RandomForestTS': 'models.random_forest_ts.spec',
    'ExtraTreesTS': 'models.extra_trees_ts.spec',
    'GradientBoostingTS': 'models.gradient_boosting_ts.spec',
    'XGBoostTS': 'models.xgboost_ts.spec',
    'LightGBMTS': 'models.lightgbm_ts.spec',
    'CatBoostTS': 'models.catboost_ts.spec',
    'ARIMATS': 'models.arima_ts.spec',
    'AutoRegressiveTS': 'models.autoregressive_ts.spec',
    'ExpSmoothingTS': 'models.exp_smoothing_ts.spec',
    'KalmanFilterTS': 'models.kalman_filter_ts.spec',
    'MLPForecasterTS': 'models.mlp_forecaster_ts.spec',
    'RNNForecasterTS': 'models.rnn_forecaster_ts.spec',
    'GRUForecasterTS': 'models.gru_forecaster_ts.spec',
    'LSTMForecasterTS': 'models.lstm_forecaster_ts.spec',
    'TCNForecasterTS': 'models.tcn_forecaster_ts.spec',
    'Aurora': 'models.aurora.spec',
    'CRIB': 'models.crib.spec',
    'TimeAlign': 'models.timealign.spec',
    'GTR': 'models.gtr.spec',
    'PhaseFormer': 'models.phaseformer.spec',
    'PMDformer': 'models.pmdformer.spec',
    'MMPD': 'models.mmpd.spec',
    'COSA': 'models.cosa.spec',
    'DistDF': 'models.distdf.spec',
    'Sonnet': 'models.sonnet.spec',
    'APN': 'models.apn.spec',
    'TimeCAP': 'models.timecap.spec',
    'GOTSF': 'models.gotsf.spec',
    'FTP': 'models.ftp.spec',
    'OccamVTS': 'models.occamvts.spec',
    'HN_MVTS': 'models.hn_mvts.spec',
    'SEMPO': 'models.sempo.spec',
    'InterPDN': 'models.interpdn.spec',
    'TimeO1': 'models.timeo1.spec',
    'FeTS': 'models.fets.spec',
    'SymTime': 'models.symtime.spec',
    'ImplicitForecaster': 'models.implicitforecaster.spec',
    'AMRC': 'models.amrc.spec',
    'HMformer': 'models.hmformer.spec',
    'TiRex': 'models.tirex.spec',
    'GlocalIB': 'models.glocalib.spec',
    'QuantileDLinear': 'models.quantile_dlinear.spec',
    'QuantilePatchTST': 'models.quantile_patchtst.spec',
    'MQRNN': 'models.mqrnn.spec',
    'GaussianMLP': 'models.gaussian_mlp.spec',
    'LatentTSF': 'models.latenttsf.spec',
    'CoRA': 'models.cora.spec',
    'DynamicTMoE': 'models.dynamic_tmoe.spec',
    'PULSE': 'models.pulse.spec',
    'OLinear': 'models.olinear.spec',
    'MAFS': 'models.mafs.spec',
    'TSRAG': 'models.tsrag.spec',
    'TimeMosaic': 'models.timemosaic.spec',
    'Kronos': 'models.kronos.spec',
    'MoFo': 'models.mofo.spec',
    'PHAT': 'models.phat.spec',
    'BiST': 'models.bist.spec',
    'MAGE': 'models.mage.spec',
    'STOP': 'models.stop.spec',
    'CauAir': 'models.cauair.spec',
    'AirCade': 'models.aircade.spec',
    'GTS': 'models.gts.spec',
    'STID': 'models.stid.spec',
    'GWNet': 'models.gwnet.spec',
    'D2STGNN': 'models.d2stgnn.spec',
    'DFDGCN': 'models.dfdgcn.spec',
    'STGCN': 'models.stgcn.spec',
    'AGCRN': 'models.agcrn.spec',
    'DCRNN': 'models.dcrnn.spec',
    'StemGNN': 'models.stemgnn.spec',
    'MTGNN': 'models.mtgnn.spec',
    'STGODE': 'models.stgode.spec',
    'STAEformer': 'models.staeformer.spec',
    'DGCRN': 'models.dgcrn.spec',
    'STDN': 'models.stdn.spec',
    'STPGNN': 'models.stpgnn.spec',
    'MegaCRN': 'models.megacrn.spec',
    'HimNet': 'models.himnet.spec',
    'STWave': 'models.stwave.spec',
    'BigST': 'models.bigst.spec',
    'ASTGCN': 'models.astgcn.spec',
    'GCLSTM': 'models.gclstm.spec',
    'DeepAir': 'models.deepair.spec',
    'STTN': 'models.sttn.spec',
    'GAGNN': 'models.gagnn.spec',
    'PM25_GNN': 'models.pm25gnn.spec',
    'AirFormer': 'models.airformer.spec',
    'DSTAGNN': 'models.dstagnn.spec',
    'PCDCNet': 'models.pcdcnet.spec',
    'AirPhyNet': 'models.airphynet.spec',
    'AirDualODE': 'models.airdualode.spec',
    'HL': 'models.hl.spec',
    'LSTM': 'models.lstm.spec',
    'RPMixer': 'models.rpmixer.spec',
    'MGSFformer': 'models.mgsfformer.spec',
    'CATS': 'models.cats.spec',
})
