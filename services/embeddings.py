import os

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


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = _get_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM
        )
    )
    vectors: list[list[float]] = [e.values for e in response.embeddings]
    return vectors
