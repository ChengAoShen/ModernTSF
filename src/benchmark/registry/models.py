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
