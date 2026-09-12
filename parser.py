import json
from dotenv import load_dotenv

from llama_cloud import LlamaCloud

load_dotenv()

client = LlamaCloud()

file = client.files.create(
    file="/home/borrelle/Working/Github Projects/folio/backend/docs_test/2206.01062v1-pages-6.pdf",
    purpose="parse"
)

result = client.parsing.parse(
    file_id=file.id, 
    tier="agentic", 
    version="latest",
    expand=["items"]
)

json_data = result.model_dump()

output_path = "/home/borrelle/Working/Github Projects/folio/backend/docs_test/resultat_parsing_2206.01062v1-pages-6.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(json_data, f, indent=4, ensure_ascii=False, default=str)


print(f"Extraction terminée! Le fichier a été sauvegarder ici: {output_path}")