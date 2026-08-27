"""Catalog for shared approximation backends used by model adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    module: str
    contract: str
    limitation: str


ADAPTER_CATALOG = {
    "recent-tsf": AdapterSpec(
        name="recent-tsf",
        module="adapters.recent_tsf",
        contract="Compact differentiable forecaster selected by explicit inductive-bias style.",
        limitation="An adaptation backend, not an implementation of any named paper model.",
    ),
}
