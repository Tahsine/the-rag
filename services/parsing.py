import os

from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv()

_client: LlamaCloud | None = None


def _get_client() -> LlamaCloud:
    global _client
    if _client is None:
        if not os.getenv("LLAMA_CLOUD_API_KEY"):
            raise RuntimeError(
                "LLAMA_CLOUD_API_KEY manquante. Ajoutez-la dans backend/.env ou dans l'environnement."
            )
        _client = LlamaCloud()
    return _client


def parse_pdf(file_path: str) -> dict:
    client = _get_client()
    file = client.files.create(file=file_path, purpose="parse")
    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["items"],
    )
    parsed_result: dict = result.model_dump()
    return parsed_result