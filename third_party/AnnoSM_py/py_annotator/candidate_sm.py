"""Map supported flavonoid structures to the substitution modes used by AnnoSM."""

from dataclasses import dataclass
import math

from rdkit import Chem


_SCAFFOLDS = {
    "Flavones": Chem.MolFromSmarts(
        "O=[c:4]1[c:3][c:2](-[c:101]2[c:102][c:103][c:104][c:105][c:106]2)"
        "[o:1][c:108]2[c:8][c:7][c:6][c:5][c:109]12"
    ),
    "IsoFlavones": Chem.MolFromSmarts(
        "O=[c:4]1[c:3](-[c:101]2[c:102][c:103][c:104][c:105][c:106]2)"
        "[c:2][o:1][c:108]2[c:8][c:7][c:6][c:5][c:109]12"
    ),
}

# Extended-library scaffolds.  The atom index sets refer to the SMARTS query
# and identify the A and B aromatic rings used by AnnoSM's mode notation.
_GENERIC_SCAFFOLDS = [
    ("Pterocarpan", "Pte", "00", "C1COc2ccccc2C1Oc1ccccc1", range(3, 9), range(11, 17)),
    ("Isoflavan", "Ifl", "00", "[CH]1([CH2]Oc2ccccc2[CH2]1)c1ccccc1", range(3, 9), range(10, 16)),
    ("Aurone", "Aur", "00", "O=C1OC(=Cc2ccccc2)c2ccccc21", range(11, 17), range(5, 11)),
    ("Anthocyanidin", "Ant", "00", "[OH+]=c1cc(-c2ccccc2)oc2ccccc12", range(11, 17), range(4, 10)),
    ("Chalcone", "Cha", "00", "O=C(/C=C/c1ccccc1)c1ccccc1", range(10, 16), range(4, 10)),
    ("Isoflavan", "Ifl", "00", "C1(-c2ccccc2)c2ccccc2OCC1", range(7, 13), range(1, 7)),
    ("Isoflavanone", "Ifn", "00", "O=C1C(-c2ccccc2)COc2ccccc21", range(11, 17), range(3, 9)),
    ("Flavanonol", "Fno", "10", "O=C1C(O)C(-c2ccccc2)Oc2ccccc21", range(12, 18), range(5, 11)),
    ("Flavan3ol", "F3o", "10", "OC1Cc2ccccc2OC1c1ccccc1", range(3, 9), range(11, 17)),
    ("Flavanone", "Fla", "00", "O=C1CC(-c2ccccc2)Oc2ccccc21", range(11, 17), range(4, 10)),
    ("Dihydrochalcone", "Dhc", "00", "O=C(CCc1ccccc1)c1ccccc1", range(10, 16), range(4, 10)),
    ("Flavan", "Fva", "00", "C1(-c2ccccc2)COc2ccccc21", range(9, 15), range(1, 7)),
    ("Xanthone", "Xan", "00", "O=c1c2ccccc2oc2ccccc12", range(2, 8), range(9, 15)),
]
_GENERIC_SCAFFOLDS = [
    (name, prefix, c_code, Chem.MolFromSmarts(smarts), tuple(a_ring), tuple(b_ring))
    for name, prefix, c_code, smarts, a_ring, b_ring in _GENERIC_SCAFFOLDS
]

_A_POSITIONS = {5, 6, 7, 8}
_B_POSITIONS = {101, 102, 103, 104, 105, 106}


@dataclass(frozen=True)
class CandidateSM:
    skeleton: str
    raw_sm: str
    annotated_sm: str
    oh_sum: int
    och3_sum: int
    a_oh_positions: tuple[int, ...]


def _mapped_atoms(query, match):
    return {
        atom.GetAtomMapNum(): match[atom.GetIdx()]
        for atom in query.GetAtoms()
        if atom.GetAtomMapNum()
    }


def _oxygen_substituent(mol, atom_idx, scaffold_atoms):
    """Return ``OH`` or ``OCH3`` for an exocyclic oxygen substituent."""
    atom = mol.GetAtomWithIdx(atom_idx)
    for neighbor in atom.GetNeighbors():
        nidx = neighbor.GetIdx()
        if nidx in scaffold_atoms or neighbor.GetSymbol() != "O":
            continue
        bond = mol.GetBondBetweenAtoms(atom_idx, nidx)
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue

        other = [n for n in neighbor.GetNeighbors() if n.GetIdx() != atom_idx]
        if not other and (
            neighbor.GetTotalNumHs() > 0 or neighbor.GetFormalCharge() == -1
        ):
            return "OH"
        if len(other) == 1 and other[0].GetSymbol() == "C":
            carbon = other[0]
            if carbon.GetTotalNumHs() == 3 and carbon.GetDegree() == 1:
                return "OCH3"

            # Acylated phenols and O-glycosides represent an OH site on the
            # aglycone. Ring ethers and arbitrary alkoxy groups do not.
            if any(
                bond.GetBondType() == Chem.BondType.DOUBLE
                and atom.GetSymbol() == "O"
                for atom in carbon.GetNeighbors()
                if (bond := mol.GetBondBetweenAtoms(carbon.GetIdx(), atom.GetIdx()))
            ):
                return "OH"
            if (
                not neighbor.IsInRing()
                and carbon.IsInRing()
                and any(
                    atom.GetSymbol() == "O" and atom.GetIdx() != neighbor.GetIdx()
                    for atom in carbon.GetNeighbors()
                )
            ):
                return "OH"
    return None


def _a_position_suffix(positions):
    positions = tuple(sorted(positions))
    if len(positions) == 1:
        return "(5/7)" if positions[0] in {5, 7} else "(6/8)"
    if len(positions) == 2:
        return "(5&7)" if set(positions) == {5, 7} else "(6/8&#)"
    return ""


def _count_ring_substituents(mol, ring_atoms, scaffold_atoms):
    oh = och3 = 0
    for atom_idx in ring_atoms:
        kind = _oxygen_substituent(mol, atom_idx, scaffold_atoms)
        if kind == "OH":
            oh += 1
        elif kind == "OCH3":
            och3 += 1
    return oh, och3


def _extract_generic_mode(mol):
    for skeleton, prefix, c_code, query, a_query_atoms, b_query_atoms in _GENERIC_SCAFFOLDS:
        match = mol.GetSubstructMatch(query)
        if not match:
            continue
        scaffold_atoms = set(match)
        a_atoms = [match[index] for index in a_query_atoms]
        b_atoms = [match[index] for index in b_query_atoms]
        a_oh, a_och3 = _count_ring_substituents(mol, a_atoms, scaffold_atoms)
        b_oh, b_och3 = _count_ring_substituents(mol, b_atoms, scaffold_atoms)
        c_oh, c_och3 = int(c_code[0]), int(c_code[1])
        raw_sm = (
            f"{prefix}A{a_oh}{a_och3}B{b_oh}{b_och3}C{c_oh}{c_och3}"
        )
        return CandidateSM(
            skeleton=skeleton,
            raw_sm=raw_sm,
            annotated_sm=raw_sm,
            oh_sum=a_oh + b_oh + c_oh,
            och3_sum=a_och3 + b_och3 + c_och3,
            a_oh_positions=(),
        )
    return None


def extract_candidate_sm(smiles):
    """Return the paper-compatible AnnoSM mode for a supported SMILES.

    The original paper covers flavones, C-3 hydroxy/methoxy flavonols and
    isoflavones. Structures outside those four parent skeletons return ``None``.
    """
    if (
        smiles is None
        or (isinstance(smiles, float) and math.isnan(smiles))
        or str(smiles).strip().lower() in {"", "nan", "none"}
    ):
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    # Match the original paper skeletons before the broader extended patterns.
    selected = None
    for skeleton in ("IsoFlavones", "Flavones"):
        query = _SCAFFOLDS[skeleton]
        match = mol.GetSubstructMatch(query)
        if match:
            selected = skeleton, query, match
            break
    if selected is None:
        return _extract_generic_mode(mol)

    skeleton, query, match = selected
    mapped = _mapped_atoms(query, match)
    scaffold_atoms = set(match)

    a_oh = a_och3 = b_oh = b_och3 = c_oh = c_och3 = 0
    a_oh_positions = []
    for position in _A_POSITIONS:
        kind = _oxygen_substituent(mol, mapped[position], scaffold_atoms)
        if kind == "OH":
            a_oh += 1
            a_oh_positions.append(position)
        elif kind == "OCH3":
            a_och3 += 1

    for position in _B_POSITIONS:
        kind = _oxygen_substituent(mol, mapped[position], scaffold_atoms)
        if kind == "OH":
            b_oh += 1
        elif kind == "OCH3":
            b_och3 += 1

    if skeleton == "Flavones":
        kind = _oxygen_substituent(mol, mapped[3], scaffold_atoms)
        if kind == "OH":
            c_oh = 1
        elif kind == "OCH3":
            c_och3 = 1

    prefix = "Iso-" if skeleton == "IsoFlavones" else ""
    a_code = f"A{a_oh}{a_och3}"
    raw_sm = f"{prefix}{a_code}B{b_oh}{b_och3}C{c_oh}{c_och3}"
    suffix = _a_position_suffix(a_oh_positions)
    annotated_a = f"{a_code}{suffix}" if suffix else a_code
    annotated_sm = f"{prefix}{annotated_a}B{b_oh}{b_och3}C{c_oh}{c_och3}"

    if skeleton == "Flavones" and c_oh:
        skeleton = "Flavonols (C-3OH)"
    elif skeleton == "Flavones" and c_och3:
        skeleton = "Flavonols (C-3OCH3)"

    return CandidateSM(
        skeleton=skeleton,
        raw_sm=raw_sm,
        annotated_sm=annotated_sm,
        oh_sum=a_oh + b_oh + c_oh,
        och3_sum=a_och3 + b_och3 + c_och3,
        a_oh_positions=tuple(sorted(a_oh_positions)),
    )
