import math

from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import SingleTurnSample

from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy
)

from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

from langchain_ollama import ChatOllama
from langchain_core.outputs import LLMResult, Generation

from src.config import (
    OLLAMA_BASE_URL,
    EVALUATOR_MODEL
)

from src.rag.embeddings import get_embeddings

# OLLAMA RAGAS LLM ADAPTER

class OllamaRagasLLM(BaseRagasLLM):

    def __init__(self):

        super().__init__()

        # RAGAS can request multiple completions.
        self.multiple_completion_supported = True

        self.llm = ChatOllama(
            model=EVALUATOR_MODEL,
            base_url=OLLAMA_BASE_URL,

            # Deterministic evaluation
            temperature=0,

            # IMPORTANT:
            # 1024 was still truncating the Faithfulness
            # statement-generation response.
            num_predict=2048,

            # Allow enough time for the cloud model.
            timeout=180
        )

    # EXTRACT TEXT FROM OLLAMA RESPONSE

    @staticmethod
    def extract_text(response):

        if response is None:
            return ""

        content = getattr(
            response,
            "content",
            None
        )

        if isinstance(content, str):

            return content.strip()

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, str):

                    parts.append(item)

                elif isinstance(item, dict):

                    if "text" in item:

                        parts.append(
                            str(item["text"])
                        )

                    elif "content" in item:

                        parts.append(
                            str(item["content"])
                        )

            return "\n".join(parts).strip()

        return ""

    # SYNCHRONOUS GENERATION

    def generate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None
    ):

        prompt_str = prompt.to_string()

        generations = []

        for i in range(n):

            print(
                f"\nGenerating completion "
                f"{i + 1}/{n}..."
            )

            response = self.llm.invoke(
                prompt_str
            )

            text = self.extract_text(
                response
            )

            print(
                "\n========== OLLAMA RESPONSE =========="
            )

            print(text)

            print(
                "======================================"
            )

            if not text:

                raise ValueError(
                    "Ollama returned an empty response."
                )

            generations.append(
                Generation(
                    text=text
                )
            )

        return LLMResult(
            generations=[
                generations
            ]
        )

    # ASYNCHRONOUS GENERATION

    async def agenerate_text(
        self,
        prompt,
        n=1,
        temperature=0.01,
        stop=None,
        callbacks=None
    ):

        prompt_str = prompt.to_string()

        generations = []

        for i in range(n):

            print(
                f"\nGenerating async completion "
                f"{i + 1}/{n}..."
            )

            response = await self.llm.ainvoke(
                prompt_str
            )

            text = self.extract_text(
                response
            )

            print(
                "\n========== OLLAMA RESPONSE =========="
            )

            print(text)

            print(
                "======================================"
            )

            if not text:

                raise ValueError(
                    "Ollama returned an empty response."
                )

            generations.append(
                Generation(
                    text=text
                )
            )

        return LLMResult(
            generations=[
                generations
            ]
        )
    # COMPLETION STATUS

    def is_finished(
        self,
        response
    ):

        return True


# EVALUATOR AGENT

def evaluator_agent(state):

    print("\n" + "=" * 60)
    print("NODE 3: EVALUATOR AGENT")
    print("=" * 60)

    question = state["question"]

    answer = state["answer"]

    contexts = state.get(
        "retrieved_context",
        []
    )

    print(
        "Running RAGAS evaluation..."
    )

    # EVALUATOR MODEL

    evaluator_llm = OllamaRagasLLM()

    # EMBEDDINGS
    evaluator_embeddings = get_embeddings()

    print(
        f"Evaluator model: "
        f"{EVALUATOR_MODEL}"
    )

    print(
        "Using local HuggingFace embeddings."
    )
    # RAGAS SAMPLE

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts
    )

    dataset = EvaluationDataset(
        samples=[
            sample
        ]
    )

    # METRICS
    metrics = [

        Faithfulness(
            llm=evaluator_llm
        ),

        ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        )
    ]
    # RUN CONFIG

    run_config = RunConfig(
        timeout=180,
        max_retries=1,
        max_workers=1,
        max_wait=5
    )

    faithfulness = None

    answer_relevancy = None

    # RUN EVALUATION

    try:

        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            run_config=run_config,
            raise_exceptions=False,
            show_progress=True
        )

        scores = result.to_pandas()

        print(
            "\nRaw RAGAS scores:"
        )

        print(scores)

        # FAITHFULNESS

        if "faithfulness" in scores.columns:

            value = scores[
                "faithfulness"
            ].iloc[0]

            try:

                value = float(value)

                if not math.isnan(value):

                    faithfulness = value

            except (
                TypeError,
                ValueError
            ):

                faithfulness = None

        # ANSWER RELEVANCY
        if "answer_relevancy" in scores.columns:

            value = scores[
                "answer_relevancy"
            ].iloc[0]

            try:

                value = float(value)

                if not math.isnan(value):

                    answer_relevancy = value

            except (
                TypeError,
                ValueError
            ):

                answer_relevancy = None

    except Exception as e:

        print(
            "\nRAGAS evaluation error:"
        )

        print(
            type(e).__name__
        )

        print(
            e
        )

    # INTERPRETATION

    if (
        faithfulness is not None
        and answer_relevancy is not None
    ):

        if (
            faithfulness >= 0.8
            and answer_relevancy >= 0.8
        ):

            interpretation = "Excellent"

        elif (
            faithfulness >= 0.6
            and answer_relevancy >= 0.6
        ):

            interpretation = "Good"

        elif (
            faithfulness >= 0.4
            or answer_relevancy >= 0.4
        ):

            interpretation = "Needs Improvement"

        else:

            interpretation = "Poor"

    else:

        interpretation = "Evaluation Incomplete"
    # RESULTS

    print(
        "\nRAGAS RESULTS"
    )

    print(
        "-" * 40
    )

    if faithfulness is None:

        print(
            "Faithfulness: N/A"
        )

    else:

        print(
            f"Faithfulness: "
            f"{faithfulness:.4f}"
        )

    if answer_relevancy is None:

        print(
            "Answer Relevancy: N/A"
        )

    else:

        print(
            f"Answer Relevancy: "
            f"{answer_relevancy:.4f}"
        )

    print(
        f"Interpretation: "
        f"{interpretation}"
    )

    # RETURN STATE


    return {

        "faithfulness": (
            faithfulness
            if faithfulness is not None
            else 0.0
        ),

        "answer_relevancy": (
            answer_relevancy
            if answer_relevancy is not None
            else 0.0
        ),

        "evaluation": {

            "faithfulness":
                faithfulness,

            "answer_relevancy":
                answer_relevancy,

            "interpretation":
                interpretation
        }
    }
