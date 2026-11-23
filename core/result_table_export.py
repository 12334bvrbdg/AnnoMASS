"""Compact public result schema, separate from internal scoring fields."""
from __future__ import annotations

import pandas as pd


ANNOTATION_BASIS = {
    "A": "A | Experimental spectral-library matching",
    "B": "B | Structure-library inference + experimental spectral evidence",
    "C": "C | Structure-library inference + predicted fragmentation evidence",
    "D": "D | Generated-structure inference + predicted fragmentation evidence",
}

EXPORT_COLUMNS = (
    "Compound name", "Formula", "SMILES", "InChIKey", "Annotation basis",
    "Candidate rank", "Annotation score", "Precursor m/z", "Spectrum IDs",
)


def compact_result_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Preserve rows/order and identifiers; expose no implementation-specific scores.

    Candidate rank and annotation score belong to the representative spectrum.
    This function never filters, deduplicates, or changes confidence assignments.
    """
    df = frame if frame is not None else pd.DataFrame()
    out = pd.DataFrame(index=df.index)

    def first(*columns):
        values = pd.Series("", index=df.index, dtype=object)
        for column in columns:
            if column in df:
                available = values.isna() | values.astype(str).str.strip().isin(("", "nan", "None"))
                values = values.where(~available, df[column])
        return values.where(values.notna(), "")

    out["Compound name"] = first("compound_name", "Compound name")
    out["Formula"] = first("formula", "Formula")
    out["SMILES"] = first("smiles", "SMILES")
    out["InChIKey"] = first("inchikey", "InChIKey")
    levels = first("confidence_level").astype(str).str.strip()
    # Older projects retain their original confidence label, without inventing
    # an A/B/C/D route from a score or a vendor-specific source string.
    out["Annotation basis"] = levels.map(ANNOTATION_BASIS).fillna(levels)
    if "Annotation basis" in df:
        out["Annotation basis"] = out["Annotation basis"].where(
            levels.ne(""), df["Annotation basis"].fillna(""))
    out["Candidate rank"] = first("spectrum_rank", "Candidate rank")
    out["Annotation score"] = first("level_score", "Annotation score")
    out["Precursor m/z"] = first("support_precursor_mz", "precursor_mz", "Precursor m/z")
    out["Spectrum IDs"] = first("spectrum_ids", "support_spectrum_titles", "spectrum_title", "Spectrum IDs")
    return out.loc[:, EXPORT_COLUMNS].copy()


def save_result_table(frame: pd.DataFrame, path) -> None:
    """Save the public table to CSV or a styled, filterable Excel sheet."""
    table = compact_result_table(frame)
    if str(path).lower().endswith(".csv"):
        table.to_csv(path, index=False, encoding="utf-8-sig")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="AnnoMASS Results", index=False)
        sheet = writer.sheets["AnnoMASS Results"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter
        widths = (34, 18, 44, 30, 68, 16, 18, 22, 38)
        for i, width in enumerate(widths, start=1):
            sheet.column_dimensions[get_column_letter(i)].width = width
            cell = sheet.cell(1, i)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="234E70")
        for row in sheet.iter_rows(min_row=2):
            for cell in row:
                # Names/identifiers are data, including strings starting '='.
                if isinstance(cell.value, str):
                    cell.data_type = "s"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
            row[6].number_format = "0.0000"
            row[7].number_format = "0.00000"
