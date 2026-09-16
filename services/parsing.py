from llama_cloud import LlamaCloud
from dotenv import load_dotenv

load_dotenv()

client: LlamaCloud = LlamaCloud()

def parse_pdf(file_path: str) -> dict:
    file = client.files.create(file=file_path, purpose="parse")
    result = client.parsing.parse(
        file_id=file.id,
        tier="agentic",
        version="latest",
        expand=["items"],
    )
    parsed_result: dict = result.model_dump()
    return parsed_result