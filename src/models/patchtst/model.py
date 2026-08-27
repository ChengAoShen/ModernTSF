"""Public PatchTST model wrapper over shared patch-transformer components."""

from components.patchtst import PatchTSTBackbone


class Model(PatchTSTBackbone):
    """Named PatchTST catalog entry; behavior is defined by ``PatchTSTBackbone``."""
