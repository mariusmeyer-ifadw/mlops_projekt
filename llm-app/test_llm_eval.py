"""
Musterlösung zu test_llm_eval.py (Tag 18) – NUR für den Kursleiter.
"""
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

import rag_utils
# Der Import von llm_app lädt dokumente.txt beim Start in Chroma (Tag 17).
from llm_app import ollama_client, CHAT_MODEL, prompt_template


def get_app_response(question: str) -> tuple[str, list[str]]:
    context_chunks = rag_utils.retrieve(question)
    context = "\n".join(context_chunks)
    prompt = prompt_template.render(context=context, question=question)

    response = ollama_client.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"], context_chunks


def test_answer_relevancy():
    # question = "Was steht in meinen Dokumenten über mein Kursprojekt?"
    question = "Womit wird die Iris-API im Kursprojekt gebaut?"
    answer, _ = get_app_response(question)

    test_case = LLMTestCase(input=question, actual_output=answer)
    metric = AnswerRelevancyMetric(threshold=0.4)

    assert_test(test_case, [metric])


'''
def test_retrieval_quality():
    # question = "Was steht in meinen Dokumenten über mein Kursprojekt?"
    question = "Womit wird die Iris-API im Kursprojekt gebaut?"
    # answer, context_chunks = get_app_response(question)
    answer, _ = get_app_response(question)
    context_chunks = rag_utils.retrieve(question, n_results=1)
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=context_chunks,
    )

    metric = ContextualRelevancyMetric(threshold=0.3)
    assert_test(test_case, [metric])
'''

def test_retrieval_quality():
    question = "Womit wird die Iris-API im Kursprojekt gebaut?"
    context_chunks = rag_utils.retrieve(question, n_results=1)

    # Regel-Check statt Judge: Die beste Fundstelle muss die FastAPI-Zeile sein
    assert "FastAPI" in context_chunks[0]
