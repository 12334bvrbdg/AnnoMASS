"""Where AnnoMASS finds its external projects and helper interpreters.

Resolution order for every item:
  1. environment variable (ANNOMASS_HOME, or a specific ANNOMASS_*_DIR / *_PYTHON)
  2. the layout created by the installer:  <ANNOMASS_HOME>/external/<Project>,
     <ANNOMASS_HOME>/envs/<env-name>
  3. the developer's legacy layout (~/<Project>, ~/anaconda3/envs/<env>)

ANNOMASS_HOME defaults to ~/AnnoMASS when that folder exists, otherwise the
legacy home directory is used, so existing installations keep working.
"""
from __future__ import annotations

import os
from pathlib import Path

_LEGACY_HOME = Path.home()


def annomass_home() -> Path:
    env = os.environ.get("ANNOMASS_HOME")
    if env:
        return Path(env).expanduser()
    new = Path.home() / "AnnoMASS"
    return new if new.is_dir() else Path.home()


def external_dir(name: str) -> Path:
    """External project folder, e.g. 'DreaMS', 'ms-pred', 'ABCoRT', 'AnnoSM', 'deepmet', 'FIDDLE'."""
    env = os.environ.get(f"ANNOMASS_{name.upper().replace('-', '_')}_DIR")
    if env:
        return Path(env).expanduser()
    for cand in (annomass_home() / "external" / name, Path.home() / name, _LEGACY_HOME / name):
        if cand.is_dir():
            return cand
    return annomass_home() / "external" / name


def env_python(env_name: str, legacy_env: str | None = None, exe: str = "python3") -> str:
    """Interpreter of a helper environment (installer name first, legacy conda env second)."""
    cands = [annomass_home() / "envs" / env_name / "bin" / exe]
    if legacy_env:
        cands += [Path.home() / "anaconda3" / "envs" / legacy_env / "bin" / exe,
                  _LEGACY_HOME / "anaconda3" / "envs" / legacy_env / "bin" / exe]
    for c in cands:
        if c.exists():
            return str(c)
    return str(cands[0])


# Environment names used by the installer (English: safe for conda/uv and paths)
ENV_MAIN = "main"
ENV_RETRIEVAL = "spectrum-retrieval"      # DreaMS reference-spectrum retrieval
ENV_PREDICTION = "spectrum-prediction"    # ICEBERG spectrum prediction
