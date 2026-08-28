"""Setuptools hook that bundles the small read-only resource catalog in wheels."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildWithRepositoryAssets(build_py):
    """Copy only resources needed for installed catalog and Agent discovery."""

    def run(self) -> None:
        # Setuptools may reuse build/lib between invocations. Start from an
        # empty build tree so deleted runtime modules cannot reappear in a
        # later wheel.
        shutil.rmtree(self.build_lib, ignore_errors=True)
        super().run()
        target = Path(self.build_lib) / "modern_tsf_assets"
        shutil.copytree(ROOT / "src" / "modern_tsf_assets", target)
        # ``egg_info`` runs before ``build_py`` and creates ``src/*.egg-info``.
        # It belongs to the distribution metadata, not to the bundled checkout
        # snapshot, so exclude it together with ordinary build detritus.
        ignored = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info", ".DS_Store"
        )
        # Runtime Python packages are already installed by build_py. The asset
        # tree contains only non-package catalogs plus src/models because cards
        # and verification fingerprints use those canonical paths. Tests and
        # maintenance scripts deliberately remain checkout-only.
        for directory in (
            ".agents",
            "catalog",
            "configs",
            "docs",
            "verification",
        ):
            shutil.copytree(
                ROOT / directory,
                target / directory,
                dirs_exist_ok=True,
                ignore=ignored,
            )
        shutil.copytree(
            ROOT / "src" / "models",
            target / "src" / "models",
            dirs_exist_ok=True,
            ignore=ignored,
        )
        for filename in (
            "AGENTS.md",
            "CHANGELOG.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "README.md",
            "README_zh.md",
            "THIRD_PARTY_NOTICES.md",
        ):
            shutil.copy2(ROOT / filename, target / filename)
        (target / ".packaged-repository").write_text(
            "read-only ModernTSF repository resources\n", encoding="utf-8"
        )


setup(cmdclass={"build_py": BuildWithRepositoryAssets})
