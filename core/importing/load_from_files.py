# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 08:43:08 2023

@author: DELL
"""

import re
import os
import tempfile
from typing import List

import matchms.filtering as msfilters
from matchms.importing import load_from_msp
from matchms.importing import load_from_mgf

from core.Spectrum import Spectrum
from core.importing.load_from_mat import load_from_mat


# Lines that pyteomics cannot parse and must be stripped before loading
_STRIP_PATTERNS = re.compile(
    r'^\s*(Num\s+Peaks|NumPeaks)\s*:',
    re.IGNORECASE,
)

PROTON = 1.00727646688
WATER = 18.010565
ADDUCT_MASS = {
    "[M+H]+": PROTON,
    "[M-H]-": -PROTON,
    "[M+NA]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M+H-H2O]+": PROTON - WATER,
    "[M+CL]-": 34.968853,
    "[M+HCOO]-": 44.997654,
    "[M+CH3COO]-": 59.013851,
    "[M+FA-H]-": 44.997654,
    "[M+AC-H]-": 59.013851,
    "[M-2H]2-": -2 * PROTON,
}


def _sanitize_mgf(path: str) -> str:
    """Return path to a cleaned temp MGF with non-standard lines removed."""
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        lines = fh.readlines()

    clean = [l for l in lines if not _STRIP_PATTERNS.match(l)]

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.mgf', delete=False, encoding='utf-8'
    )
    tmp.writelines(clean)
    tmp.close()
    return tmp.name


_MSP_NUM_PEAKS = re.compile(r'^\s*(num\s*peaks|number\s+of\s+peaks|npeaks)\s*[:=]\s*\d*', re.IGNORECASE)
_MSP_PEAK_LINE = re.compile(r'^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?[\s,;:]+[-+]?\d')


def _sanitize_msp(path: str) -> str:
    """Return path to a normalised temp MSP.

    matchms' MSP reader needs a ``Num Peaks:`` line in every record. MSP files from different tools spell it
    differently (``NumPeaks``, ``NUM PEAKS``, ``Number of peaks``) or omit it, and the whole file then fails to
    load. This rewrites every record with a standard ``Num Peaks: <n>`` line (n = the actual number of peak
    lines), normalises line endings and leaves every other line untouched."""
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        text = fh.read().replace('\r\n', '\n').replace('\r', '\n')
    out = []
    block = []

    def flush():
        if not any(l.strip() for l in block):
            block.clear()
            return
        meta = [l for l in block if not _MSP_NUM_PEAKS.match(l) and not _MSP_PEAK_LINE.match(l) and l.strip()]
        peaks = [l for l in block if _MSP_PEAK_LINE.match(l)]
        out.extend(meta)
        out.append(f"Num Peaks: {len(peaks)}")
        for l in peaks:
            out.append(re.sub(r'[,;:]+', ' ', l.strip()))      # "mz;int" / "mz,int" / "mz:int" -> "mz int"
        out.append("")
        block.clear()

    for line in text.split('\n'):
        if line.strip() == '':
            flush()
        else:
            block.append(line)
    flush()
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.msp', delete=False, encoding='utf-8')
    tmp.write('\n'.join(out) + '\n')
    tmp.close()
    return tmp.name


def clean_spectrum(s):
    s = msfilters.default_filters(s)
    s = msfilters.add_compound_name(s)
    s = msfilters.correct_charge(s)
    s = msfilters.add_parent_mass(s)
    s = msfilters.normalize_intensities(s)
    s = harmonize_ion_metadata(s)
    return s


def _clean_adduct(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in ("", "nan", "none", "null"):
        return ""
    return text


def _canonical_adduct(value: object) -> str:
    text = _clean_adduct(value)
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text).upper()
    aliases = {
        "M-H": "[M-H]-",
        "[M-H]": "[M-H]-",
        "[M-H]-": "[M-H]-",
        "M+H": "[M+H]+",
        "[M+H]": "[M+H]+",
        "[M+H]+": "[M+H]+",
        "[M+NA]+": "[M+Na]+",
        "[M+K]+": "[M+K]+",
        "[M+NH4]+": "[M+NH4]+",
        "[M+H-H2O]+": "[M+H-H2O]+",
        "[M-H2O+H]+": "[M+H-H2O]+",
        "[M+CL]-": "[M+Cl]-",
        "[M+HCOO]-": "[M+HCOO]-",
        "[M+HCOOH-H]-": "[M+HCOO]-",
        "[M+CH3COO]-": "[M+CH3COO]-",
        "[M+CH3COOH-H]-": "[M+CH3COO]-",
        "[M+FA-H]-": "[M+FA-H]-",
        "[M+AC-H]-": "[M+AC-H]-",
        "[M-2H]2-": "[M-2H]2-",
    }
    return aliases.get(compact, text)


def _infer_ionmode_from_adduct(adduct: str) -> str:
    if not adduct:
        return ""
    compact = re.sub(r"\s+", "", adduct).upper()
    if compact.endswith("-") or "-]" in compact or "-H]" in compact or "-H-" in compact:
        return "negative"
    if compact.endswith("+") or "+]" in compact:
        return "positive"
    return ""


def _format_charge_for_ionmode(ionmode: str, existing: object):
    adduct_charge = _infer_charge_from_adduct(existing)
    if adduct_charge:
        return adduct_charge
    if ionmode == "negative":
        return -1
    if ionmode == "positive":
        return 1
    return existing


def _infer_charge_from_adduct(adduct: object):
    compact = re.sub(r"\s+", "", str(adduct or "")).upper()
    m = re.search(r"\](\d*)([+-])$", compact)
    if not m:
        return None
    z = int(m.group(1) or "1")
    return z if m.group(2) == "+" else -z


def _neutral_mass_from_adduct(precursor_mz, adduct: str, charge=None):
    try:
        mz = float(precursor_mz)
    except Exception:
        return None
    key = _canonical_adduct(adduct).upper()
    add = ADDUCT_MASS.get(key)
    if add is None:
        return None
    z = abs(int(charge or _infer_charge_from_adduct(adduct) or 1))
    return mz * max(z, 1) - add


def harmonize_ion_metadata(s):
    """Fix ion mode and neutral mass when adduct contradicts MGF CHARGE.

    Some GNPS/MGF exports encode negative spectra as ``ION=[M-H]-`` while
    keeping ``CHARGE=1+``. matchms then marks them positive and computes an
    incorrect parent mass. Downstream DeepMASS model selection is ionmode-based,
    so adduct evidence has to win over the charge sign.
    """
    adduct = (
        s.get("adduct") or s.get("ion") or s.get("precursor_type") or
        s.get("ADDUCT") or s.get("ION") or s.get("PRECURSOR_TYPE") or ""
    )
    adduct = _canonical_adduct(adduct)
    ionmode = _infer_ionmode_from_adduct(adduct)
    if not ionmode:
        raw_ionmode = str(s.get("ionmode") or s.get("IONMODE") or "").strip().lower()
        if raw_ionmode.startswith("neg"):
            ionmode = "negative"
        elif raw_ionmode.startswith("pos"):
            ionmode = "positive"

    if adduct:
        s.set("adduct", adduct)
        s.set("ion", adduct)
    if ionmode:
        s.set("ionmode", ionmode)
        s.set("charge", _format_charge_for_ionmode(ionmode, adduct or s.get("charge")))

    parent_mass = _neutral_mass_from_adduct(s.get("precursor_mz"), adduct, s.get("charge"))
    if parent_mass is not None:
        s.set("parent_mass", parent_mass)
    return s


def load_from_files(filenames: List[str]) -> List[Spectrum]:
    output = []
    for filename in filenames:
        patt = filename.split('.')[-1].lower()
        if patt == 'msp':
            tmp_path = _sanitize_msp(filename)
            try:
                spectrums = load_from_msp(tmp_path)
                for s in spectrums:
                    s = clean_spectrum(s)
                    output.append(Spectrum(mz=s.mz,
                                           intensities=s.intensities,
                                           metadata=s.metadata))
            finally:
                os.unlink(tmp_path)
        elif patt == 'mgf':
            tmp_path = _sanitize_mgf(filename)
            try:
                spectrums = load_from_mgf(tmp_path)
                for s in spectrums:
                    s = clean_spectrum(s)
                    output.append(Spectrum(mz=s.mz,
                                           intensities=s.intensities,
                                           metadata=s.metadata))
            finally:
                os.unlink(tmp_path)
        elif patt == 'mat':
            spectrums = load_from_mat(filename)
            for s in spectrums:
                isotopic_mz = s.isotopic_pattern.mz
                isotopic_intensities = s.isotopic_pattern.intensities
                s = clean_spectrum(s)
                output.append(Spectrum(mz=s.mz,
                                       intensities=s.intensities,
                                       isotopic_mz=isotopic_mz,
                                       isotopic_intensities=isotopic_intensities,
                                       metadata=s.metadata))
        else:
            continue

    output_1 = []
    for s in output:
        if s.get('compound_name') is None:
            s.set('compound_name', 'Unknown_{}'.format(len(output_1)))
        output_1.append(s)
    return output_1
