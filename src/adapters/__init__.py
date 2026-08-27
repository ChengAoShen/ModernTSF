"""Explicit shared adaptation backends.

Adapters are separated from reusable mathematical components because they
represent acknowledged approximation strategies rather than paper-neutral
building blocks.
"""

from adapters.catalog import ADAPTER_CATALOG, AdapterSpec

__all__ = ["ADAPTER_CATALOG", "AdapterSpec"]
