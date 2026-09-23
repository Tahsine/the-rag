"""Unit test for bbox=None handling in chunking.

Why some items have no bbox:
- LlamaParse agentic tier returns items where bbox can be None, missing, or an empty list.
  Observed on `docs_test/2206.01062v1.pdf` (9 pages, full doc) — some `type` items (headings skipped, but also
  text/table items) came back with `"bbox": null` (Python None) or `"bbox": []` or with entries missing x/y/w/h.
  The old code did `for b in item["bbox"]` without guard → TypeError: 'NoneType' object is not iterable.
- We want to keep the text even when geometry is missing: chunk with `sources=[Source(item_index, bbox=[])]`
  still participates in RAG (snippet + page) but just won't have a highlight rectangle. The bbox is optional for
  retrieval/generation; citation_images already handles empty bbox gracefully (skip drawing).

Run: python test_chunking_bbox.py  (no API keys needed)
"""

from services.chunking import build_chunks
from schemas import Chunk

def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)

def test_bbox_none():
    seen: set[str] = set()
    items = [
        {"type": "text", "md": "Hello world", "bbox": None},
        {"type": "text", "md": "Another text", "bbox": []},
        {"type": "text", "md": "Missing bbox key", },  # no bbox key
        {"type": "text", "md": "List with None entries", "bbox": [None, {"x": 10, "y": 10, "w": 100, "h": 10}, None]},
        {"type": "text", "md": "Incomplete bbox", "bbox": [{"x": 10, "y": 10}, {"x": 1, "y": 1, "w": 10, "h": 10, "confidence": 0.9}]},
        {"type": "heading", "md": "Heading without bbox", "bbox": None},  # should be skipped, pending_heading set
        {"type": "text", "md": "Text after heading", "bbox": None},
        {"type": "text", "md": "   ", "bbox": None},  # empty text -> skipped
        {"type": "text", "md": None, "bbox": None},  # None text -> skipped
    ]
    chunks: list[Chunk] = build_chunks(items, page_number=1, doc_id="test-doc", seen_texts=seen)
    # Expected: heading skipped, empty text skipped, None text skipped → 6 chunks? Let's count
    # - Hello world (bbox None -> []) : 1
    # - Another text ([] -> []): 2
    # - Missing bbox key (None -> []): 3
    # - List with None entries (one valid bbox): 4
    # - Incomplete bbox (one valid of two): 5
    # - heading skipped
    # - Text after heading (heading prefix, bbox None -> []): 6
    # - "   " skipped
    # - None skipped
    _assert(len(chunks) == 6, f"expected 6 chunks, got {len(chunks)}: {[c.text for c in chunks]}")
    # Check bbox handling
    _assert(chunks[0].sources[0].bbox == [], f"bbox None should give []: {chunks[0].sources[0].bbox}")
    _assert(chunks[1].sources[0].bbox == [], "bbox [] should stay []")
    _assert(chunks[2].sources[0].bbox == [], "missing bbox key should give []")
    _assert(len(chunks[3].sources[0].bbox) == 1, f"None entries filtered, expected 1 valid bbox, got {chunks[3].sources[0].bbox}")
    _assert(len(chunks[4].sources[0].bbox) == 1, "incomplete bbox filtered")
    # Text after heading should have heading prefix and empty bbox
    _assert(chunks[5].text.startswith("Heading without bbox:"), f"heading prefix missing: {chunks[5].text}")
    _assert(chunks[5].sources[0].bbox == [], "heading-follow text with None bbox -> []")
    print("✓ test_bbox_none passed")

def test_bbox_normal_still_works():
    seen: set[str] = set()
    items = [
        {"type": "text", "md": "Normal", "bbox": [{"x": 1, "y": 2, "w": 3, "h": 4}]},
    ]
    chunks = build_chunks(items, page_number=2, doc_id="d1", seen_texts=seen)
    _assert(len(chunks) == 1, "normal bbox should give 1 chunk")
    _assert(len(chunks[0].sources[0].bbox) == 1, "normal bbox length 1")
    _assert(chunks[0].sources[0].bbox[0].x == 1, "bbox x")
    print("✓ test_bbox_normal_still_works passed")

def test_dedup_and_heading():
    seen: set[str] = set()
    items = [
        {"type": "text", "md": "Hello", "bbox": None},
        {"type": "text", "md": "hello", "bbox": None},  # duplicate normalized -> skipped
        {"type": "heading", "md": "Chap 1", "bbox": None},
        {"type": "text", "md": "Body", "bbox": None},
    ]
    chunks = build_chunks(items, page_number=1, doc_id="d1", seen_texts=seen)
    _assert(len(chunks) == 2, f"dedup+heading: expected 2, got {len(chunks)}")
    _assert(chunks[1].text == "Chap 1: Body", f"heading prefix: {chunks[1].text}")
    print("✓ test_dedup_and_heading passed")

if __name__ == "__main__":
    test_bbox_none()
    test_bbox_normal_still_works()
    test_dedup_and_heading()
    print("\nAll chunking bbox tests passed.")
