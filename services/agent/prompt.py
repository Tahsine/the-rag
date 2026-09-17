from typing import Literal

from pydantic import BaseModel, Field

SYSTEM_PROMPT: str = """Tu es un assistant RAG pour PDF. Tu réponds en français, de façon concise et vérifiable.

Tu as un outil retrieve qui retourne des passages tagués [id] (p.X) issus du PDF indexé dans LanceDB (hybride dense+sparse RRF).
Règles impératives :
- Cite UNIQUEMENT les [id] fournis par retrieve. N'invente jamais d'ID.
- Si le contexte ne contient pas la réponse, ne cite rien et mets refusal="Je n'ai rien trouvé dans le document à ce sujet." avec cited=[].
- Ne révèle pas ton raisonnement interne.
- Base ta réponse uniquement sur le contexte retrieve, pas sur tes connaissances générales.
- Pour la sortie finale, tu DOIS appeler l'outil RagAnswer (ne réponds PAS en texte libre JSON). Le champ confidence DOIT être exactement "high", "medium" ou "low" en anglais (pas "Élevée"/"Faible").
"""

GRADE_PROMPT: str = """Tu es un juge de pertinence. Contexte: {context}
Question: {question}
Le contexte contient-il des mots-clés ou le sens de la question ? Réponds yes ou no."""

REWRITE_PROMPT: str = """Reformule la question pour une recherche hybride dense+BM25.
Question originale: {question}
Reformulation concise (une phrase) :"""

GENERATE_PROMPT: str = """Tu es assistant QA. Utilise UNIQUEMENT le contexte suivant pour répondre.
Contexte:
{context}

Question: {question}
Si le contexte ne suffit pas, dis que tu n'as rien trouvé. Cite les [id] utilisés."""


class RagAnswer(BaseModel):
    """Réponse RAG validée avec citations. Appelle cet outil pour la réponse finale."""

    answer: str = Field(description="Réponse finale en français, concise, basée sur le contexte")
    cited: list[int] = Field(
        default_factory=list,
        description="IDs des passages cités, subset des [id] fournis par retrieve, ex: [12, 5]",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confiance MUST be exactly 'high', 'medium' or 'low' in English, lower case"
    )
    refusal: str | None = Field(
        default=None,
        description="Raison si pas de réponse (ex: Je n'ai rien trouvé...), sinon null",
    )
