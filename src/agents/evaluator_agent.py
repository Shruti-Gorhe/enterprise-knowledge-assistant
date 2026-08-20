from typing import Optional, List

from ollama import Client
from langchain_core.outputs import LLMResult, Generation

from ragas.llms.base import BaseRagasLLM
from ragas.metrics.collections import Faithfulness, AnswerRelevancy

from src.config import (
    OLLAMA_BASE_URL,
    EVALUATOR_MODEL,
)


class OllamaRagasLLM(BaseRagasLLM):

    def __init__(self):
        super().__init__()

        self.client = Client(
            host=OLLAMA_BASE_URL
        )

    def generate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:

        if hasattr(prompt, "to_string"):
            prompt_text = prompt.to_string()
        else:
            prompt_text = str(prompt)

        response = self.client.chat(
            model=EVALUATOR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt_text,
                }
            ],
            options={
                "temperature": temperature,
            },
        )

        text = response["message"]["content"]

        return LLMResult(
            generations=[
                [
                    Generation(
                        text=text
                    )
                ]
            ]
        )

    async def agenerate_text(
        self,
        prompt,
        n: int = 1,
        temperature: float = 0.01,
        stop: Optional[List[str]] = None,
        callbacks=None,
    ) -> LLMResult:

        return self.generate_text(
            prompt=prompt,
            n=n,
            temperature=temperature,
            stop=stop,
            callbacks=callbacks,
        )

    def is_finished(
        self,
        response: LLMResult
    ) -> bool:

        return True


def evaluator_agent(state):

    print("\n" + "=" * 60)
    print("NODE 3: EVALUATOR AGENT")
    print("=" * 60)

    print("Running RAGAS evaluation...")

    question = state["question"]
    answer = state["answer"]
    contexts = state.get(
        "retrieved_context",
        []
    )

    try:

        ragas_llm = OllamaRagasLLM()

        faithfulness_metric = Faithfulness(
            llm=ragas_llm
        )

        relevancy_metric = AnswerRelevancy(
            llm=ragas_llm
        )

        from ragas import EvaluationDataset, evaluate

        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                }
            ]
        )

        result = evaluate(
            dataset,
            metrics=[
                faithfulness_metric,
                relevancy_metric,
            ],
        )

        result_df = result.to_pandas()

        faithfulness_score = float(
            result_df["faithfulness"].iloc[0]
        )

        relevancy_score = float(
            result_df["answer_relevancy"].iloc[0]
        )

        print("\nRAGAS RESULTS")
        print("-" * 40)

        print(
            f"Faithfulness: "
            f"{faithfulness_score:.4f}"
        )

        print(
            f"Answer Relevancy: "
            f"{relevancy_score:.4f}"
        )

    except Exception as e:

        print("\nRAGAS evaluation error:")
        print(type(e).__name__)
        print(e)

        faithfulness_score = 0.0
        relevancy_score = 0.0

    average_score = (
        faithfulness_score +
        relevancy_score
    ) / 2

    if average_score >= 0.8:
        interpretation = "Excellent"

    elif average_score >= 0.6:
        interpretation = "Good"

    elif average_score >= 0.4:
        interpretation = "Needs Improvement"

    else:
        interpretation = "Poor"

    print("\nEVALUATION RESULTS")
    print("-" * 40)

    print(
        f"Faithfulness: "
        f"{faithfulness_score:.4f}"
    )

    print(
        f"Answer Relevancy: "
        f"{relevancy_score:.4f}"
    )

    print(
        f"Interpretation: "
        f"{interpretation}"
    )

    return {
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "evaluation": {
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "interpretation": interpretation,
        },
    }