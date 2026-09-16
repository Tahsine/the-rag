from google import genai
from google.genai import types
from dotenv import load_dotenv

from config import EMBEDDING_MODEL, EMBEDDING_DIM

load_dotenv()

client: genai.Client = genai.Client()

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBEDDING_DIM
        )
    )
    vectors: list[list[float]] = [e.values for e in response.embeddings]
    return vectors
