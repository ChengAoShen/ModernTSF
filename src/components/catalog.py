"""Flat metadata catalog for reusable model components.

The catalog documents shared implementation contracts without creating a
second model hierarchy.  It is metadata only: models import the concrete
component modules directly, so catalog discovery never changes runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ComponentSpec:
    """Identity and semantic boundary of one reusable component module."""

    name: str
    module: str
    contract: str
    public_symbols: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentMatch:
    """One retrieval candidate; semantic compatibility still needs review."""

    spec: ComponentSpec
    score: int
    matched_terms: tuple[str, ...]


class ComponentCatalog:
    """Flat lookup for shared component metadata."""

    def __init__(self, specs: tuple[ComponentSpec, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("component names must be unique")

    def names(self) -> list[str]:
        return sorted(self._specs)

    def get(self, name: str) -> ComponentSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown component {name!r}") from exc

    def specs(self) -> tuple[ComponentSpec, ...]:
        return tuple(self._specs[name] for name in self.names())

    def match(self, query: str, limit: int = 5) -> tuple[ComponentMatch, ...]:
        """Rank lexical candidates without claiming semantic equivalence."""
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        if not terms or limit < 1:
            return ()
        matches: list[ComponentMatch] = []
        for spec in self.specs():
            name_terms = set(re.findall(r"[a-z0-9]+", spec.name.lower()))
            symbol_terms = set(
                re.findall(r"[a-z0-9]+", " ".join(spec.public_symbols).lower())
            )
            keyword_terms = set(
                re.findall(r"[a-z0-9]+", " ".join(spec.keywords).lower())
            )
            contract_terms = set(re.findall(r"[a-z0-9]+", spec.contract.lower()))
            matched = terms & (
                name_terms | symbol_terms | keyword_terms | contract_terms
            )
            if not matched:
                continue
            score = sum(
                5
                if term in name_terms
                else 3
                if term in keyword_terms
                else 2
                if term in symbol_terms
                else 1
                for term in matched
            )
            matches.append(ComponentMatch(spec, score, tuple(sorted(matched))))
        matches.sort(key=lambda item: (-item.score, item.spec.name))
        return tuple(matches[:limit])


COMPONENT_CATALOG = ComponentCatalog(
    (
        ComponentSpec("adj_norm", "components.adj_norm", "Dense adjacency normalization.", keywords=("adjacency", "graph", "laplacian", "normalization")),
        ComponentSpec("auto_correlation", "components.auto_correlation", "Auto-correlation attention layers.", keywords=("autocorrelation", "attention", "periodicity", "seasonality")),
        ComponentSpec("autoformer_encdec", "components.autoformer_encdec", "Series decomposition and Autoformer encoder/decoder blocks.", keywords=("autoformer", "decomposition", "encoder", "decoder")),
        ComponentSpec("base", "components.base", "Minimal dimensional base class for adapted upstream models.", ("BaseModel",), ("adapter", "upstream", "shape")),
        ComponentSpec("conv_blocks", "components.conv_blocks", "Reusable temporal convolution blocks.", keywords=("convolution", "temporal", "kernel")),
        ComponentSpec(
            "dlinear",
            "components.dlinear",
            "Moving-average decomposition and channel-wise linear forecasting backbone.",
            ("MovingAvg", "SeriesDecomp", "DLinearBackbone"),
            ("decomposition", "linear", "moving-average", "seasonal", "trend"),
        ),
        ComponentSpec("embed", "components.embed", "Value, position, calendar, patch, and inverted embeddings.", keywords=("calendar", "embedding", "patch", "position", "token")),
        ComponentSpec(
            "flatten_forecast_head",
            "components.flatten_forecast_head",
            "Shared or channel-wise linear forecast head over two flattened feature axes.",
            ("FlattenForecastHead",),
            ("channel-wise", "flatten", "forecast", "head", "linear", "patch"),
        ),
        ComponentSpec("fourier_correlation", "components.fourier_correlation", "Fourier-domain correlation layers.", keywords=("correlation", "fourier", "frequency", "spectral")),
        ComponentSpec("graph_utils", "components.graph_utils", "Graph supports, Laplacians, and Chebyshev bases.", keywords=("adjacency", "chebyshev", "graph", "laplacian", "support")),
        ComponentSpec("marks", "components.marks", "Canonical temporal-mark and spatiotemporal input adapters.", keywords=("calendar", "covariate", "spatiotemporal", "timestamp")),
        ComponentSpec(
            "mamba",
            "components.mamba",
            "Kernel-free selective state-space mixer, normalization, and residual block.",
            ("RMSNorm", "MambaBlock", "MambaResidualBlock"),
            ("mamba", "mixer", "rmsnorm", "ssm", "state-space"),
        ),
        ComponentSpec("masking", "components.masking", "Attention mask construction.", keywords=("attention", "causal", "mask")),
        ComponentSpec(
            "patchtst",
            "components.patchtst",
            "Patch extraction, time-series Transformer encoding, and PatchTST backbone.",
            ("FlattenHead", "iTSTEncoder", "PatchTSTModel", "PatchTSTBackbone"),
            ("backbone", "channel-independent", "patch", "transformer"),
        ),
        ComponentSpec("positional_encoding", "components.positional_encoding", "Patch-transformer positional encodings.", keywords=("encoding", "patch", "position", "transformer")),
        ComponentSpec(
            "quantile_head",
            "components.quantile_head",
            "Input-conditioned monotone quantile head with non-crossing outputs.",
            ("QuantileHead",),
            ("monotone", "non-crossing", "probabilistic", "quantile"),
        ),
        ComponentSpec("revin", "components.revin", "Reversible instance normalization.", ("RevIN",), ("denormalization", "instance", "normalization", "reversible")),
        ComponentSpec("self_attention_family", "components.self_attention_family", "Shared full and probabilistic attention layers.", keywords=("attention", "full", "probabilistic")),
        ComponentSpec(
            "series_decomposition",
            "components.series_decomposition",
            "Edge-padded moving average and residual/trend decomposition for BLC data.",
            ("EdgePaddedMovingAverage", "SeriesDecomposition"),
            ("decomposition", "moving-average", "residual", "smoothing", "trend"),
        ),
        ComponentSpec("standard_norm", "components.standard_norm", "Normalize/de-normalize transform used by forecasting models.", keywords=("denormalization", "forecast", "normalization")),
        ComponentSpec("transformer_encdec", "components.transformer_encdec", "Shared Transformer encoder and decoder blocks.", keywords=("attention", "decoder", "encoder", "transformer")),
        ComponentSpec("tst_transformer", "components.tst_transformer", "Time-series Transformer encoder blocks.", keywords=("attention", "encoder", "time-series", "transformer")),
    )
)
