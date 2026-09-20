from typing import Literal

from pydantic import BaseModel, Field

SYSTEM_PROMPT: str = """Tu es un assistant RAG pour PDF. Tu réponds en français, de façon concise et vérifiable.

Tu as un outil retrieve qui retourne des passages tagués [id] (doc:X p.Y) issus du corpus indexé dans LanceDB (hybride dense+sparse RRF).
Le retrieve peut être limité à un seul document ou porter sur tout le corpus, selon la demande.
Règles impératives :
- Cite UNIQUEMENT les [id] fournis par retrieve. N'invente jamais d'ID.
- Si le contexte ne contient pas la réponse, ne cite rien et mets refusal="Je n'ai rien trouvé dans le corpus à ce sujet." avec cited=[].
- Ne révèle pas ton raisonnement interne.
- Base ta réponse uniquement sur le contexte retrieve, pas sur tes connaissances générales.
- Si plusieurs documents sont cités, précise le document concerné quand c'est utile.
- Pour la sortie finale, tu DOIS appeler l'outil RagAnswer (ne réponds PAS en texte libre JSON). Le champ confidence DOIT être exactement "high", "medium" ou "low" en anglais (pas "Élevée"/"Faible").
"""

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
