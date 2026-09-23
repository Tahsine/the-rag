import os
import re
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

from config import EMBEDDING_MODEL, EMBEDDING_DIM

load_dotenv()

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError(
                "GOOGLE_API_KEY manquante. Ajoutez-la dans backend/.env ou dans l'environnement."
            )
        _client = genai.Client()
    return _client


def _extract_retry_delay(err: Exception) -> float | None:
    msg = str(err)
    # cherche "retry in 51.090075973s" ou "retryDelay': '51s'"
    m = re.search(r"retry in ([\d.]+)s", msg)
    if m:
        try:
            return float(m.group(1)) + 1.0
        except ValueError:
            pass
    m2 = re.search(r"retryDelay.*?(\d+)s", msg)
    if m2:
        try:
            return float(m2.group(1)) + 1.0
        except ValueError:
            pass
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "Quota exceeded" in msg:
        return 60.0
    return None


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    # Gemini BatchEmbedContents is limited to 100 texts per batch + 100 req/min free tier
    batch_size = 90  # marge sous 100
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # retry avec backoff pour quota 429
        last_err: Exception | None = None
        for attempt in range(4):
            try:
                response = _get_client().models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=EMBEDDING_DIM
                    )
                )
                all_vectors.extend([e.values for e in response.embeddings])
                last_err = None
                break
            except Exception as e:
                last_err = e
                delay = _extract_retry_delay(e)
                if delay is not None and attempt < 3:
                    time.sleep(delay)
                    continue
                raise
        if last_err is not None:
            raise last_err
        # petite pause entre batches pour ne pas enchainer 100 req/min
        if i + batch_size < len(texts):
            time.sleep(0.5)
    return all_vectors
