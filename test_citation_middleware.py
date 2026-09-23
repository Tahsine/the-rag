def test_citation_validation_filters_when_nothing_retrieved():
    from services.agent.middleware.citation import CitationValidationMiddleware
    from services.agent.prompt import RagAnswer

    middleware = CitationValidationMiddleware()
    fake_state = {
        "structured_response": RagAnswer(answer="...", cited=[3, 7], confidence="high", refusal=None),
        "messages": [],
        "retrieved_ids": [],
    }
    result = middleware.after_model(fake_state, runtime=None)

    assert result is not None
    assert result["structured_response"].cited == []
    assert result["structured_response"].confidence == "low"
    print("test passé")

if __name__ == "__main__":
    test_citation_validation_filters_when_nothing_retrieved()