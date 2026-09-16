from services.embeddings import embed_texts

texts: list[str] = ["Hello world", "A List-item is a paragraph with hanging indentation."]
vectors: list[list[float]] = embed_texts(texts)

print(f"{len(vectors)} vecteurs générés")
print(f"dimension de chaque vecteur: {len(vectors[0])}")
assert len(vectors) == len(texts), "un vecteur par texte, pas un de moins/plus"