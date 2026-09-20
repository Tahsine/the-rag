from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv()

_client: LlamaCloud | None = None


def _get_client() -> LlamaCloud:
    global _client
    if _client is None:
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