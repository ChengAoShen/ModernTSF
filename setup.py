"""Setuptools hook that bundles a read-only repository snapshot in wheels."""

from __future__ import annotations

from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent


class BuildWithRepositoryAssets(build_py):
    """Copy canonical non-package resources beside the installed Python API."""

    def run(self) -> None:
        super().run()
        target = Path(self.build_lib) / "modern_tsf_assets"
        # A wheel rebuild may reuse ``build/lib``. Recreate the snapshot so a
        # file removed from the repository (or excluded below) cannot survive
        # from an earlier build.
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(ROOT / "src" / "modern_tsf_assets", target)
        # ``egg_info`` runs before ``build_py`` and creates ``src/*.egg-info``.
        # It belongs to the distribution metadata, not to the bundled checkout
        # snapshot, so exclude it together with ordinary build detritus.
        ignored = shutil.ignore_patterns(
            "__pycache__", "*.pyc", "*.egg-info", ".DS_Store"
        )
        for directory in (
            ".agents",
            "catalog",
            "configs",
            "docs",
            "scripts",
            "tests",
            "verification",
        ):
            shutil.copytree(
                ROOT / directory,
                target / directory,
                dirs_exist_ok=True,
                ignore=ignored,
            )
        shutil.copytree(
            ROOT / "src",
            target / "src",
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
