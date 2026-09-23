from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import pymupdf as fitz
from PIL import Image, ImageDraw

from config import CITATION_IMAGES_DIR, CITATION_IMAGE_ZOOM
from schemas import CitationImage
from services.agent.prompt import RagAnswer
from services.citations import iter_cited_rows, parse_sources
from services.document_store import has_source_pdf, stored_pdf_path, validate_doc_id

_PALETTE: list[tuple[int, int, int]] = [
    (239, 68, 68),
    (59, 130, 246),
    (34, 197, 94),
    (234, 179, 8),
    (168, 85, 247),
    (236, 72, 153),
    (20, 184, 166),
    (249, 115, 22),
]


@dataclass
class CitationImageResult:
    images: list[CitationImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _iter_bbox_rects(row: Any):
    for source in parse_sources(row):
        bboxes = source.get("bbox")
        if not isinstance(bboxes, list):
            continue
        for bbox in bboxes:
            if not isinstance(bbox, dict):
                continue
            try:
                x = float(bbox.get("x", 0))
                y = float(bbox.get("y", 0))
                w = float(bbox.get("w", 0))
                h = float(bbox.get("h", 0))
            except (TypeError, ValueError):
                continue
            if w <= 0 or h <= 0:
                continue
            yield (x, y, x + w, y + h)


def save_citation_images(
    structured: RagAnswer | None,
    doc_id: str | None = None,
    answer_id: str | None = None,
) -> CitationImageResult:
    """Génère un PNG par page citée, avec toutes les bboxes des chunks cités."""
    result = CitationImageResult()
    if not structured or not structured.cited:
        return result

    answer_id = answer_id or uuid.uuid4().hex[:8]
    pages: dict[tuple[str, int], dict[str, Any]] = {}

    for color_index, (cid, row) in enumerate(iter_cited_rows(structured, doc_id)):
        raw_doc_id = str(row.get("doc_id", "") or "")
        try:
            safe_doc_id = validate_doc_id(raw_doc_id)
        except ValueError:
            result.warnings.append(f"Image citation ignorée: doc_id non sûr {raw_doc_id!r}")
            continue
        try:
            page = int(row.get("page", 1))
        except (TypeError, ValueError):
            result.warnings.append(f"Image citation ignorée: page invalide pour le chunk {cid}")
            continue

        entry = pages.setdefault(
            (safe_doc_id, page),
            {"doc_id": safe_doc_id, "page": page, "boxes": []},
        )
        color = _PALETTE[color_index % len(_PALETTE)]
        for rect in _iter_bbox_rects(row):
            entry["boxes"].append((rect, f"c{cid}", color))

    for entry in pages.values():
        entry_doc_id: str = entry["doc_id"]
        entry_page: int = entry["page"]
        if not has_source_pdf(entry_doc_id):
            result.warnings.append(
                f"Image citation non générée pour {entry_doc_id} p.{entry_page}: PDF source absent "
                f"(utilisez `just-rag attach-pdf {entry_doc_id} <pdf>`)"
            )
            continue

        try:
            pdf_path = stored_pdf_path(entry_doc_id)
            with fitz.open(pdf_path) as pdf:
                if entry_page < 1 or entry_page > len(pdf):
                    result.warnings.append(
                        f"Image citation non générée pour {entry_doc_id} p.{entry_page}: page absente du PDF source"
                    )
                    continue
                page = pdf[entry_page - 1]
                pix = page.get_pixmap(matrix=fitz.Matrix(CITATION_IMAGE_ZOOM, CITATION_IMAGE_ZOOM))
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                draw = ImageDraw.Draw(img)

                bbox_count = 0
                for (x0, y0, x1, y1), label, color in entry["boxes"]:
                    rect = [x0 * CITATION_IMAGE_ZOOM, y0 * CITATION_IMAGE_ZOOM, x1 * CITATION_IMAGE_ZOOM, y1 * CITATION_IMAGE_ZOOM]
                    draw.rectangle(rect, outline=color, width=3)
                    draw.text((rect[0] + 2, rect[1] + 2), label, fill=color)
                    bbox_count += 1

                if bbox_count == 0:
                    result.warnings.append(
                        f"Image citation non générée pour {entry_doc_id} p.{entry_page}: aucune bbox valide"
                    )
                    continue

                out_dir = CITATION_IMAGES_DIR / entry_doc_id / answer_id
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"page_{entry_page:03d}.png"
                img.save(out_path)
                result.images.append(
                    CitationImage(
                        doc_id=entry_doc_id,
                        page=entry_page,
                        path=str(out_path),
                        bbox_count=bbox_count,
                    )
                )
        except Exception as e:
            result.warnings.append(f"Image citation non générée pour {entry_doc_id} p.{entry_page}: {e}")

    return result
