from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pandas as pd


def open_pdf(pdf_path: str | Path) -> fitz.Document:
    """
    Open a PDF file with PyMuPDF.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file.

    Returns
    -------
    fitz.Document
        Opened PDF document.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return fitz.open(pdf_path)


def get_page(doc: fitz.Document, page_number: int) -> fitz.Page:
    """
    Get a page from an open PDF document.

    Parameters
    ----------
    doc : fitz.Document
        Open PDF.
    page_number : int
        Zero-indexed page number.

    Returns
    -------
    fitz.Page
        Requested page.
    """
    if page_number < 0 or page_number >= len(doc):
        raise IndexError(f"Page {page_number} out of bounds for document with {len(doc)} pages.")
    return doc[page_number]


def extract_drawings(page: fitz.Page) -> list[dict[str, Any]]:
    """
    Extract raw vector drawing objects from a PDF page.

    Parameters
    ----------
    page : fitz.Page
        PDF page.

    Returns
    -------
    list[dict]
        Raw drawing dictionaries from PyMuPDF.
    """
    return page.get_drawings()


def flatten_line_segments(drawings: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Extract line-segment primitives from PyMuPDF drawings.

    Only items of type 'l' are retained. Each row corresponds to one
    vector line segment with geometric and style metadata.

    Parameters
    ----------
    drawings : list[dict]
        Raw drawing objects from page.get_drawings().

    Returns
    -------
    pd.DataFrame
        DataFrame of line segments.
    """
    records: list[dict[str, Any]] = []

    for drawing_idx, drawing in enumerate(drawings):
        items = drawing.get("items", [])
        stroke_color = drawing.get("color")
        width = drawing.get("width")
        layer = drawing.get("layer")
        seqno = drawing.get("seqno")
        rect = drawing.get("rect")

        for item_idx, item in enumerate(items):
            if not item:
                continue

            item_type = item[0]
            if item_type != "l":
                continue

            p0 = item[1]
            p1 = item[2]

            x0, y0 = float(p0.x), float(p0.y)
            x1, y1 = float(p1.x), float(p1.y)

            records.append(
                {
                    "drawing_idx": drawing_idx,
                    "item_idx": item_idx,
                    "item_type": item_type,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "dx": x1 - x0,
                    "dy": y1 - y0,
                    "length": ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5,
                    "is_horizontal": abs(y1 - y0) < 1e-8,
                    "is_vertical": abs(x1 - x0) < 1e-8,
                    "stroke_color": stroke_color,
                    "width": width,
                    "layer": layer,
                    "seqno": seqno,
                    "rect_x0": float(rect.x0) if rect is not None else None,
                    "rect_y0": float(rect.y0) if rect is not None else None,
                    "rect_x1": float(rect.x1) if rect is not None else None,
                    "rect_y1": float(rect.y1) if rect is not None else None,
                }
            )

    return pd.DataFrame(records)


def extract_line_segments_from_pdf(
    pdf_path: str | Path,
    page_number: int,
) -> pd.DataFrame:
    """
    Open a PDF, extract one page, and return flattened line segments.

    Parameters
    ----------
    pdf_path : str or Path
        PDF path.
    page_number : int
        Zero-indexed page number.

    Returns
    -------
    pd.DataFrame
        Line-segment DataFrame.
    """
    doc = open_pdf(pdf_path)
    try:
        page = get_page(doc, page_number)
        drawings = extract_drawings(page)
        segments = flatten_line_segments(drawings)
    finally:
        doc.close()

    if not segments.empty:
        segments["page_number"] = page_number
        segments["pdf_path"] = str(Path(pdf_path))

    return segments


def extract_text_blocks(page: fitz.Page) -> pd.DataFrame:
    """
    Extract text blocks from a PDF page.

    Useful for later axis-label or risk-table parsing.

    Parameters
    ----------
    page : fitz.Page

    Returns
    -------
    pd.DataFrame
        Text blocks with coordinates.
    """
    blocks = page.get_text("blocks")
    records: list[dict[str, Any]] = []

    for idx, block in enumerate(blocks):
        x0, y0, x1, y1, text, block_no, block_type = block[:7]
        records.append(
            {
                "block_idx": idx,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "text": text,
                "block_no": block_no,
                "block_type": block_type,
            }
        )

    return pd.DataFrame(records)


def extract_text_blocks_from_pdf(
    pdf_path: str | Path,
    page_number: int,
) -> pd.DataFrame:
    """
    Open a PDF and extract text blocks from one page.
    """
    doc = open_pdf(pdf_path)
    try:
        page = get_page(doc, page_number)
        text_df = extract_text_blocks(page)
    finally:
        doc.close()

    if not text_df.empty:
        text_df["page_number"] = page_number
        text_df["pdf_path"] = str(Path(pdf_path))

    return text_df


def summarize_page_graphics(
    pdf_path: str | Path,
    page_number: int,
) -> dict[str, Any]:
    """
    Quick summary of vector content on a page.

    Returns counts of drawings, line segments, and text blocks.
    """
    doc = open_pdf(pdf_path)
    try:
        page = get_page(doc, page_number)
        drawings = extract_drawings(page)
        segments = flatten_line_segments(drawings)
        text_blocks = extract_text_blocks(page)
    finally:
        doc.close()

    return {
        "pdf_path": str(Path(pdf_path)),
        "page_number": page_number,
        "n_drawings": len(drawings),
        "n_line_segments": int(len(segments)),
        "n_horizontal_segments": int(segments["is_horizontal"].sum()) if not segments.empty else 0,
        "n_vertical_segments": int(segments["is_vertical"].sum()) if not segments.empty else 0,
        "n_text_blocks": int(len(text_blocks)),
    }