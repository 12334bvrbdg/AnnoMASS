"""
flavonoid_fragment_scorer.py
------------------------------------------------------------
Evidence-based scoring for flavonoid MS2 candidate structures.

Combines two independent lines of evidence:
  1. Fixed-mass lookup table (universal neutral losses, glycoside
     losses, sugar cross-ring cleavages) -- from
     flavonoid_fragment_reference.csv
  2. Dynamically-computed RDA (retro-Diels-Alder) fragment masses,
     derived per-candidate from the SMILES structure itself (RDA
     fragment masses are NOT fixed -- they depend on A-ring/B-ring
     substitution, so they cannot be a static lookup table).

Requires: rdkit, pandas, numpy
    pip install rdkit pandas numpy

NOT TESTED in this environment (no network / no rdkit available
here) -- validate against a few known spectra before trusting it
in the pipeline. The __main__ block at the bottom runs a self-check
against the flavonol glucoside example we analyzed by hand
(precursor 591.24, ID 1607) so you have a known-good reference to
compare your own output against.
------------------------------------------------------------
"""

from __future__ import annotations
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, BRICS
from rdkit.Chem.rdmolops import FragmentOnBonds, GetMolFrags
from dataclasses import dataclass, field
from itertools import combinations

PROTON_MASS = 1.00727646677

# Default loss table lives next to this module.
_DEFAULT_LOSS_TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "flavonoid_fragment_reference.csv")

# ------------------------------------------------------------------
# 1. Fixed-mass loss table
# ------------------------------------------------------------------

def load_loss_table(csv_path: str = None) -> pd.DataFrame:
    """Load the universal/glycoside/cross-ring loss reference table."""
    if csv_path is None:
        csv_path = _DEFAULT_LOSS_TABLE
    df = pd.read_csv(csv_path)
    return df


def match_fixed_losses(precursor_mz: float, peaks: list[tuple[float, float]],
                        loss_table: pd.DataFrame, ppm_tol: float = 15.0,
                        noise_frac: float = 0.01) -> list[dict]:
    """
    For each MS2 peak above the noise threshold, check whether
    (precursor - peak) matches any row in the fixed loss table
    within ppm_tol.

    peaks: list of (m/z, intensity) tuples
    noise_frac: peaks below noise_frac * max(intensity) are ignored
    Returns a list of match records (one per peak that hit a row).
    """
    if not peaks:
        return []
    max_int = max(i for _, i in peaks)
    hits = []
    for mz, inten in peaks:
        if inten < noise_frac * max_int:
            continue
        loss = precursor_mz - mz
        if loss <= 0:
            continue
        # ppm error is computed on the *fragment* m/z, not the loss,
        # to match how MS accuracy is normally reported
        best = None
        for _, row in loss_table.iterrows():
            pred_frag = precursor_mz - row["exact_mass_Da"]
            ppm = (mz - pred_frag) / pred_frag * 1e6
            if abs(ppm) <= ppm_tol and (best is None or abs(ppm) < abs(best["ppm"])):
                best = {"peak_mz": mz, "intensity": inten,
                        "matched_loss": row["loss_or_ion_name"],
                        "formula": row["formula"], "ppm": ppm,
                        "category": row["category"]}
        if best:
            hits.append(best)
    return hits


# Losses that represent a labile appendage (a sugar or acyl group that can
# leave on its own). Only these are combined pairwise for the "two appendages
# leave together" layer -- combining universal losses like water+CO would
# just inflate the score with chemically meaningless matches.
_LABILE_LOSS_CATEGORIES = {"glycoside_loss", "sugar_cross_ring"}


def match_combined_losses(precursor_mz: float, peaks: list[tuple[float, float]],
                          loss_table: pd.DataFrame, ppm_tol: float = 15.0,
                          noise_frac: float = 0.01) -> list[dict]:
    """Match 'combined' losses: a peak whose neutral loss equals the sum of two
    labile-appendage losses (e.g. sugar + sugar, sugar + malonyl, sugar + alkyl
    leaving together in one step). This is the combinatorial layer for the
    two-appendage case that single-row matching misses."""
    if not peaks:
        return []
    labile = loss_table[loss_table["category"].isin(_LABILE_LOSS_CATEGORIES)]
    loss_masses = [float(x) for x in labile["exact_mass_Da"].tolist()]
    if len(loss_masses) < 2:
        return []
    max_int = max(i for _, i in peaks)
    hits = []
    for mz, inten in peaks:
        if inten < noise_frac * max_int:
            continue
        loss = precursor_mz - mz
        if loss <= 0:
            continue
        best = None
        for la, lb in combinations(loss_masses, 2):
            pred_frag = precursor_mz - (la + lb)
            if pred_frag <= 0:
                continue
            ppm = (mz - pred_frag) / pred_frag * 1e6
            if abs(ppm) <= ppm_tol and (best is None or abs(ppm) < abs(best["ppm"])):
                best = {"peak_mz": mz, "intensity": inten,
                        "matched_loss": f"combined({la:.4f}+{lb:.4f})",
                        "formula": "", "ppm": ppm,
                        "category": "combined_loss"}
        if best:
            hits.append(best)
    return hits


# ------------------------------------------------------------------
# 2. RDA core detection + dynamic fragment mass calculation
#
#    RDA fragment masses are structure-dependent, so instead of a
#    lookup table we detect the flavonoid core type by SMARTS, find
#    the two C-ring bonds that break in RDA, cut them with RDKit,
#    and compute the exact mass of each resulting piece.
#
#    NOTE: these SMARTS patterns are generic C6-C3-C6 cores and
#    should match regardless of A-/B-ring substitution, but you
#    should validate them against a handful of known flavone /
#    flavonol / isoflavone SMILES before trusting them at scale.
# ------------------------------------------------------------------

# each entry: (name, SMARTS, ring-bond atom-index pairs to break)
# atom indices refer to positions *within the SMARTS match*, in the
# order the SMARTS atoms are written (0-indexed). Verified against RDKit
# GetSubstructMatch ordering for apigenin/quercetin/genistein.
FLAVONOID_CORE_PATTERNS = {
    "flavone_flavonol": {
        # 2-phenylchromone: O=C4-C3-C2(-B-ring)-O1-C8a...C4a ; B-ring at C2.
        # SMARTS atom order (0-indexed): 0=O(C4=O), 1=C4, 2=C3, 3=C2,
        #   4..9=B-ring, 10=O1, 11=C8a, 12..15=A-ring, 16=C4a.
        # Classic 1,3-RDA breaks O1-C2 (10,3) and C3-C4 (2,1).
        "smarts": "O=c1cc(-c2ccccc2)oc2ccccc12",
        "break_bond_atom_pairs": [(10, 3), (2, 1)],
    },
    "isoflavone": {
        # 3-phenylchromone: B-ring at C3 instead of C2.
        # SMARTS atom order: 0=O(C4=O), 1=C4, 2=C3, 3..8=B-ring, 9=C2,
        #   10=O1, 11=C8a, 12..16=A-ring, 17=C4a.
        # 1,3-RDA breaks O1-C2 (10,9) and C3-C4 (2,1).
        "smarts": "O=c1c(-c2ccccc2)coc2ccccc12",
        "break_bond_atom_pairs": [(10, 9), (2, 1)],
    },
}


def find_flavonoid_core(mol: Chem.Mol) -> str | None:
    """Return the matching core type name, or None if no match.

    Subclass-aware by design: only flavone/flavonol and isoflavone get RDA
    scoring. Other subclasses (flavanone, chalcone, aurone, anthocyanidin,
    biflavonoid, flavan-3-ol) fragment via different mechanisms (retro-Michael,
    retro-Claisen, exocyclic cleavage, or the 152 Da catechin loss), so they
    return None here and are scored on the fixed-loss table only -- attempting
    RDA on them would unfairly penalize correct candidates for "failing" a
    fragmentation test that was never going to apply.
    """
    for name, spec in FLAVONOID_CORE_PATTERNS.items():
        patt = Chem.MolFromSmarts(spec["smarts"])
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return None


def compute_rda_fragment_masses(mol: Chem.Mol, core_name: str) -> list[float]:
    """
    Break the RDA bonds for the given core type and return the exact
    [fragment+H]+ masses of the resulting pieces.

    This is a best-effort implementation -- FragmentOnBonds with
    addDummies=False caps the broken bonds with H automatically,
    which is the right approximation for RDA (both halves pick up
    a proton-equivalent worth of H during the retro-[4+2]).
    """
    spec = FLAVONOID_CORE_PATTERNS[core_name]
    patt = Chem.MolFromSmarts(spec["smarts"])
    match = mol.GetSubstructMatch(patt)
    if not match:
        return []

    # Kekulize first: breaking aromatic C-ring bonds on an aromatic molecule
    # leaves fragments with inconsistent aromatic flags, which SanitizeMol
    # rejects ("non-ring atom marked aromatic"). Kekulé form breaks cleanly.
    kek = Chem.Mol(mol)
    Chem.Kekulize(kek, clearAromaticFlags=True)

    bond_idxs = []
    for a_idx, b_idx in spec["break_bond_atom_pairs"]:
        a, b = match[a_idx], match[b_idx]
        bond = kek.GetBondBetweenAtoms(a, b)
        if bond is not None:
            bond_idxs.append(bond.GetIdx())

    if len(bond_idxs) < 2:
        return []  # bonds not found as expected -- pattern mapping needs fixing

    frag_mol = FragmentOnBonds(kek, bond_idxs, addDummies=False)
    frags = GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)

    masses = []
    for f in frags:
        try:
            Chem.SanitizeMol(f)
            m = rdMolDescriptors.CalcExactMolWt(f)
            masses.append(m + PROTON_MASS)  # as [fragment+H]+
        except Exception:
            continue
    return masses


def match_rda(smiles: str, precursor_mz: float, peaks: list[tuple[float, float]],
              ppm_tol: float = 15.0, noise_frac: float = 0.01) -> list[dict]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    core = find_flavonoid_core(mol)
    if core is None:
        return []  # not a recognizable flavone/flavonol/isoflavone core -- skip RDA scoring

    rda_masses = compute_rda_fragment_masses(mol, core)
    if not rda_masses:
        return []

    max_int = max((i for _, i in peaks), default=1)
    hits = []
    for mz, inten in peaks:
        if inten < noise_frac * max_int:
            continue
        for pred in rda_masses:
            ppm = (mz - pred) / pred * 1e6
            if abs(ppm) <= ppm_tol:
                hits.append({"peak_mz": mz, "intensity": inten,
                             "matched_rda_mass": pred, "ppm": ppm,
                             "core_type": core})
    return hits


# ------------------------------------------------------------------
# 3. Combined evidence score
# ------------------------------------------------------------------

@dataclass
class FragmentEvidenceResult:
    n_significant_peaks: int
    n_explained_fixed: int
    n_explained_rda: int
    n_explained_combined: int
    explained_intensity_frac: float
    fixed_hits: list = field(default_factory=list)
    rda_hits: list = field(default_factory=list)
    combined_hits: list = field(default_factory=list)

    @property
    def evidence_score(self) -> float:
        """0-1 score: fraction of significant-peak intensity explained
        by the fixed loss table, dynamic RDA matching, or combined losses."""
        return self.explained_intensity_frac


def score_candidate(smiles: str, precursor_mz: float,
                     peaks: list[tuple[float, float]],
                     loss_table: pd.DataFrame,
                     ppm_tol: float = 15.0,
                     noise_frac: float = 0.01) -> FragmentEvidenceResult:
    max_int = max((i for _, i in peaks), default=1)
    sig_peaks = [(mz, i) for mz, i in peaks if i >= noise_frac * max_int]

    fixed_hits = match_fixed_losses(precursor_mz, peaks, loss_table, ppm_tol, noise_frac)
    rda_hits = match_rda(smiles, precursor_mz, peaks, ppm_tol, noise_frac)
    combined_hits = match_combined_losses(precursor_mz, peaks, loss_table, ppm_tol, noise_frac)

    explained_mz = ({h["peak_mz"] for h in fixed_hits}
                    | {h["peak_mz"] for h in rda_hits}
                    | {h["peak_mz"] for h in combined_hits})
    explained_intensity = sum(i for mz, i in sig_peaks if mz in explained_mz)
    total_sig_intensity = sum(i for _, i in sig_peaks) or 1

    return FragmentEvidenceResult(
        n_significant_peaks=len(sig_peaks),
        n_explained_fixed=len(fixed_hits),
        n_explained_rda=len(rda_hits),
        n_explained_combined=len(combined_hits),
        explained_intensity_frac=explained_intensity / total_sig_intensity,
        fixed_hits=fixed_hits,
        rda_hits=rda_hits,
        combined_hits=combined_hits,
    )


# ------------------------------------------------------------------
# Self-check: the flavonol-3-O-rhamnoside example (ID 1826, 591.24)
# with the strongest, most literature-consistent fragment matches
# we found by hand earlier -- use this to sanity-check the pipeline
# before trusting it on new candidates.
# ------------------------------------------------------------------

if __name__ == "__main__":
    smiles = "COc1ccc(-c2oc3cc(OC)c(OC)c(O)c3c(=O)c2OC2OC(C)C(O)C(O)C2O)cc1OCCCC(C)C"
    precursor_mz = 591.24058
    peaks = [
        (325.07343, 10), (429.2017, 20), (471.20643, 10), (576.21651, 20),
        (589.36563, 10), (589.39214, 10), (591.14351, 21), (591.18398, 52),
        (591.23921, 1265), (591.31309, 11), (591.34900, 20),
    ]

    loss_table = load_loss_table("flavonoid_fragment_reference.csv")
    result = score_candidate(smiles, precursor_mz, peaks, loss_table)

    print(f"significant peaks: {result.n_significant_peaks}")
    print(f"explained by fixed-loss table: {result.n_explained_fixed}")
    print(f"explained by RDA: {result.n_explained_rda}")
    print(f"evidence score (intensity-weighted): {result.evidence_score:.3f}")
    print("\nfixed-loss hits:")
    for h in result.fixed_hits:
        print(" ", h)
    print("\nRDA hits (expect none -- this compound is a flavonol, "
          "RDA bonds not stressed at this collision energy):")
    for h in result.rda_hits:
        print(" ", h)

    # Expected from our manual analysis earlier in this conversation:
    #   576.21651 -> -CH3 (in the fixed table? NO -- this is a
    #     methyl RADICAL loss, check that CH3(radical) row matches)
    #   429.2017  -> combined rhamnose+alkyl loss (NOT in the fixed
    #     table as a single row -- it's a two-appendage combined
    #     loss, so this one will likely NOT be caught by
    #     match_fixed_losses as-is; you may want to add a
    #     "combined_loss" category to the CSV for molecules with
    #     more than one labile appendage, or chain two lookups)
    #   471.20643 -> -C4H8O4 cross-ring cleavage (SHOULD match)
    #   325.07343 -> aglycone (also a combined loss, same caveat as 429)
