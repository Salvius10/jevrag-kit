import importlib
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

import jevrag_kit


@pytest.mark.parametrize("module", ["jevrag_kit", "jevrag_kit.classifier", "jevrag_kit.llm", "jevrag_kit.checker"])
def test_every_exported_name_exists(module):
    mod = importlib.import_module(module)
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert missing == []


def test_no_export_hides_a_submodule():
    # `from jevrag_kit.checker import locate` must not make `import jevrag_kit.checker.<module>` return a function.
    import pkgutil

    shadowed = []
    for pkg_name in ("jevrag_kit", "jevrag_kit.classifier", "jevrag_kit.llm", "jevrag_kit.checker"):
        pkg = importlib.import_module(pkg_name)
        for info in pkgutil.iter_modules(pkg.__path__):
            if info.name != "__main__" and getattr(pkg, info.name) is not importlib.import_module(f"{pkg_name}.{info.name}"):
                shadowed.append(f"{pkg_name}.{info.name}")
    assert shadowed == []


def test_installed_version_matches_the_package():
    assert version("jevrag-kit") == jevrag_kit.__version__


def test_package_data_is_present():
    root = Path(jevrag_kit.__file__).parent
    assert (root / "py.typed").exists() and (root / "default_config.yaml").exists()


def test_importing_the_package_does_not_import_provider_sdks():
    code = (
        "import sys, jevrag_kit, jevrag_kit.classifier, jevrag_kit.llm, jevrag_kit.checker, jevrag_kit.engine, jevrag_kit.replay, jevrag_kit.cli\n"
        "print(sorted(m for m in ('typesafe_sdk', 'anthropic', 'openai') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
