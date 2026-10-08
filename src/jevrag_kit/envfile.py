"""Load API keys from a `.env` file into the environment, without extra dependencies."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import MutableMapping

from jevrag_kit.errors import ConfigError

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_env_file(path: str | Path = ".env", environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Load KEY=VALUE lines from `path` into `environ` (default: os.environ).

    Comments (#), blank lines, and lines that are not assignments are skipped. `export KEY=...`,
    quoted values, and trailing ` # comments` are understood. Variables that are already set are
    kept, so a real environment variable always wins over the file. Returns the names loaded.
    """
    path = Path(path)
    env = os.environ if environ is None else environ
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read the env file ({exc.strerror or exc})") from exc
    loaded = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not _NAME.fullmatch(key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if key not in env:
            env[key] = value
            loaded.append(key)
    return loaded
