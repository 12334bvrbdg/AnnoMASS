#!/usr/bin/env bash
# AnnoMASS one-shot installer for Linux / WSL2-Ubuntu.
# Usage:  bash install.sh            # from inside the downloaded AnnoMASS folder
#   ANNOMASS_HOME=/data/AnnoMASS bash install.sh     # choose another install location
#   ANNOMASS_SKIP_DOWNLOAD=1 bash install.sh          # do not download model/data files
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${HERE}/../.." && pwd)"                 # downloaded AnnoMASS folder (has launch_gui.py)
export ANNOMASS_HOME="${ANNOMASS_HOME:-${HOME}/AnnoMASS}"
EXT="${ANNOMASS_HOME}/external"                        # DreaMS, ms-pred, ABCoRT, AnnoSM, deepmet, FIDDLE
ENVS="${ANNOMASS_HOME}/envs"                           # main / spectrum-retrieval / spectrum-prediction
CONDA="${ANNOMASS_HOME}/conda"
log()  { printf '\n[AnnoMASS] %s\n' "$*"; }
fail() { printf '\n[AnnoMASS] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "${APP_DIR}/launch_gui.py" ]] || fail "launch_gui.py not found; run install.sh from the AnnoMASS folder."
command -v git  >/dev/null || fail "git is required (sudo apt install git)."
command -v curl >/dev/null || fail "curl is required (sudo apt install curl)."
command -v nvidia-smi >/dev/null || log "WARNING: nvidia-smi not found. Only the NVIDIA GPU path has been validated."
mkdir -p "${EXT}" "${ENVS}"

# 1. conda (miniforge) ---------------------------------------------------------------
if [[ ! -x "${CONDA}/bin/conda" ]]; then
  log "Installing Miniforge into ${CONDA}"
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "${CONDA}" && rm -f /tmp/miniforge.sh
fi
export PATH="${CONDA}/bin:${PATH}"
command -v uv >/dev/null || "${CONDA}/bin/pip" install --quiet uv

# 2. Three environments (cannot be merged: DreaMS pins numpy 1.25.0, ICEBERG needs numpy>=2) --
mk_env() {   # prefix, yml
  if [[ -x "$1/bin/python" ]]; then log "env $(basename "$1") exists"; return; fi
  log "Creating environment $(basename "$1") (this takes a while)"
  "${CONDA}/bin/conda" env create -p "$1" -f "$2"
}
mk_env "${ENVS}/main"                "${HERE}/envs/env_deepmass2.yml"   # main program, DeepMASS, ranking
mk_env "${ENVS}/spectrum-retrieval"  "${HERE}/envs/env_dreams.yml"      # DreaMS reference-spectrum retrieval

# 3. External projects pinned to the commits used for the paper, plus local patches ----
clone_pin() {  # name url commit patch
  local d="${EXT}/$1"
  if [[ ! -d "${d}/.git" ]]; then log "Cloning $1"; git clone --quiet "$2" "${d}"; fi
  git -C "${d}" checkout --quiet "$3"
  if [[ -n "${4:-}" && -f "${HERE}/patches/$4" ]] && git -C "${d}" apply --check "${HERE}/patches/$4" 2>/dev/null; then
    git -C "${d}" apply "${HERE}/patches/$4"; log "patched $1"
  fi
}
clone_pin DreaMS  https://github.com/pluskal-lab/DreaMS.git e37a47d DreaMS.patch
clone_pin ms-pred https://github.com/coleygroup/ms-pred.git  c35bf82 ms-pred.patch
clone_pin ABCoRT  https://github.com/RiverCCC/ABCoRT.git     a68e4ff ABCoRT.patch
clone_pin AnnoSM  https://github.com/YangHuaLab/AnnoSM.git   3be9b3a ""
# Python port of AnnoSM used by AnnoMASS (not part of the upstream repository)
rm -rf "${EXT}/AnnoSM/py_annotator" && cp -r "${APP_DIR}/third_party/AnnoSM_py/py_annotator" "${EXT}/AnnoSM/"

# 4. Spectrum-prediction (ICEBERG) environment ----------------------------------------
if [[ ! -x "${ENVS}/spectrum-prediction/bin/python" ]]; then
  log "Creating spectrum-prediction environment"
  uv venv --python 3.11 "${ENVS}/spectrum-prediction"
  uv pip install -p "${ENVS}/spectrum-prediction/bin/python" \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -f https://data.pyg.org/whl/torch-2.11.0+cu128.html \
      -r "${HERE}/envs/ms-pred_requirements.txt"
  ( cd "${EXT}/ms-pred" && uv pip install -p "${ENVS}/spectrum-prediction/bin/python" -e . )
fi

# 5. Model weights, spectral libraries and structure libraries (GitHub Release, about 4 GB) ---
if [[ "${ANNOMASS_SKIP_DOWNLOAD:-0}" != 1 ]]; then
  log "Downloading model and data files"
  bash "${HERE}/download_data.sh" || log "WARNING: download incomplete; run download_data.sh again later."
fi
check_files() {   # path-relative-to-ANNOMASS_HOME or APP_DIR ...
  local missing=0
  while read -r base rel; do
    [[ -z "${base}" || "${base}" == \#* ]] && continue
    root="${ANNOMASS_HOME}"; [[ "${base}" == APP ]] && root="${APP_DIR}"
    [[ -e "${root}/${rel}" ]] || { printf '  missing: %s/%s\n' "${root}" "${rel}"; missing=1; }
  done < "${HERE}/required_files.txt"
  return ${missing}
}
log "Checking model/data files (see WEIGHTS.md for where to get them)"
if check_files; then log "All model/data files present."; else
  log "Some model/data files are missing. Run: bash ${HERE}/download_data.sh  (or see WEIGHTS.md)"
fi

# 6. Entry points ---------------------------------------------------------------------
install -d "${HOME}/.local/bin" "${HOME}/.local/share/applications"
sed -e "s|@ANNOMASS_HOME@|${ANNOMASS_HOME}|g" -e "s|@APP_DIR@|${APP_DIR}|g" "${HERE}/annomass" > "${HOME}/.local/bin/annomass"
chmod 755 "${HOME}/.local/bin/annomass"
sed "s|@APP_DIR@|${APP_DIR}|g" "${HERE}/annomass.desktop.in" > "${HOME}/.local/share/applications/annomass.desktop"
log "Done.  Start with:  annomass     (or the AnnoMASS entry in your application menu)"
