#!/usr/bin/env bash
# Download the AnnoMASS model weights, spectral libraries and structure libraries from the
# GitHub Release and put each file where AnnoMASS looks for it. Files that are already in place
# are skipped, so the script can be run again after an interrupted download.
#
# Usage:  bash release_files/installer/download_data.sh
#   ANNOMASS_HOME=/data/AnnoMASS bash release_files/installer/download_data.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${HERE}/../.." && pwd)"
export ANNOMASS_HOME="${ANNOMASS_HOME:-${HOME}/AnnoMASS}"
RELEASE="${ANNOMASS_RELEASE:-v1.0.0}"
BASE_URL="${ANNOMASS_DATA_URL:-https://github.com/12334bvrbdg/AnnoMASS/releases/download/${RELEASE}}"
log() { printf '[AnnoMASS] %s\n' "$*"; }

command -v curl >/dev/null || { echo "curl is required (sudo apt install curl)." >&2; exit 1; }

# release file                                         base  target folder
FILES="
references_index_positive_spec2vec.bin                  APP   data
references_index_negative_spec2vec.bin                  APP   data
references_spectrums_positive.pickle                    APP   data
references_spectrums_negative.pickle                    APP   data
MassSpecGym_DreaMS_flavonoid_only.hdf5                  APP   data/reference_libraries
spec2vec_spectraverse_positive.model                    APP   model
spec2vec_spectraverse_positive.model.syn1neg.npy        APP   model
spec2vec_spectraverse_positive.model.wv.vectors.npy     APP   model
spec2vec_spectraverse_negative.model                    APP   model
spec2vec_spectraverse_negative.model.syn1neg.npy        APP   model
spec2vec_spectraverse_negative.model.wv.vectors.npy     APP   model
FlavonoidStructureDB-dedup.csv                          APP   Stru
flavonoid_candidates_final.csv                          HOME  external/deepmet/final_results
CFIL_extended.csv                                       HOME  external/AnnoSM/Library
TPIL_extended.xlsx                                      HOME  external/AnnoSM/Library
ssl_model.ckpt                                          HOME  external/DreaMS/dreams/models/pretrained
embedding_model.ckpt                                    HOME  external/DreaMS/dreams/models/pretrained
"

fetch() {  # name target_dir
  local name="$1" dir="$2"
  if [[ -s "${dir}/${name}" ]]; then log "present: ${name}"; return; fi
  mkdir -p "${dir}"
  log "downloading ${name}"
  curl -fL --retry 3 -C - -o "${dir}/${name}.part" "${BASE_URL}/${name}"
  mv "${dir}/${name}.part" "${dir}/${name}"
}

while read -r name base rel; do
  [[ -z "${name}" ]] && continue
  root="${ANNOMASS_HOME}"; [[ "${base}" == APP ]] && root="${APP_DIR}"
  fetch "${name}" "${root}/${rel}"
done <<< "${FILES}"

# ICEBERG weights come as one zip that holds the folder msg_all/
ICEBERG_DIR="${ANNOMASS_HOME}/external/ms-pred/models"
if [[ -s "${ICEBERG_DIR}/msg_all/gen/best.ckpt" && -s "${ICEBERG_DIR}/msg_all/inten_contr/best.ckpt" ]]; then
  log "present: msg_all (ICEBERG)"
else
  fetch msg_all.zip "${ICEBERG_DIR}"
  PY="${ANNOMASS_HOME}/conda/bin/python"; [[ -x "${PY}" ]] || PY=python3
  ( cd "${ICEBERG_DIR}" && "${PY}" -m zipfile -e msg_all.zip . )
  rm -f "${ICEBERG_DIR}/msg_all.zip"
  log "unpacked msg_all (ICEBERG)"
fi

log "All files downloaded."
