from ollama import Client

from src.config import (
    OLLAMA_BASE_URL,
    EVALUATOR_MODEL,
)


def evaluator_agent(state):

    print("\n" + "=" * 60)
    print("NODE 3: EVALUATOR AGENT")
    print("=" * 60)

    print("Running local evaluation...")

    question = state["question"]
    answer = state["answer"]
    contexts = state.get("retrieved_context", [])

    context_text = "\n\n---\n\n".join(contexts)

    prompt = f"""
You are an evaluation agent for a Retrieval-Augmented Generation system.

Evaluate the generated answer using ONLY the supplied context.

Evaluate:

1. FAITHFULNESS
Does the answer contain claims that are supported by the context?

2. RELEVANCY
Does the answer directly answer the user's question?

Give both scores between 0 and 1.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context_text}

GENERATED ANSWER:
{answer}

Return ONLY:

FAITHFULNESS: 0.90
RELEVANCY: 0.90

Replace the example numbers with your actual scores.
"""

    try:

        client = Client(
            host=OLLAMA_BASE_URL
        )

        response = client.chat(
            model=EVALUATOR_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={
                "temperature": 0,
            },
        )

        raw_output = response["message"]["content"]

        print("\nRaw evaluator output:")
        print(repr(raw_output))

        faithfulness = None
        relevancy = None

        for line in raw_output.splitlines():

            line = line.strip().upper()

            if line.startswith("FAITHFULNESS:"):

                try:
                    faithfulness = float(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError:
                    pass

            elif line.startswith("RELEVANCY:"):

                try:
                    relevancy = float(
                        line.split(":", 1)[1].strip()
                    )
                except ValueError:
                    pass

        if faithfulness is None:
            faithfulness = 0.0

        if relevancy is None:
            relevancy = 0.0

        faithfulness = max(
            0.0,
            min(1.0, faithfulness)
        )

        relevancy = max(
            0.0,
            min(1.0, relevancy)
        )

    except Exception as e:

        print("\nEvaluator error:")
        print(e)

        faithfulness = 0.0
        relevancy = 0.0

    # ------------------------------------------------
    # INTERPRETATION
    # ------------------------------------------------

    if (
        faithfulness >= 0.8
        and relevancy >= 0.8
    ):
        interpretation = "Excellent"

    elif (
        faithfulness >= 0.6
        and relevancy >= 0.6
    ):
        interpretation = "Good"

    elif (
        faithfulness >= 0.4
        or relevancy >= 0.4
    ):
        interpretation = "Needs Improvement"

    else:
        interpretation = "Poor"

    # ------------------------------------------------
    # RESULTS
    # ------------------------------------------------

    print("\nEVALUATION RESULTS")
    print("-" * 40)

    print(
        f"Faithfulness: "
        f"{faithfulness:.4f}"
    )

    print(
        f"Answer Relevancy: "
        f"{relevancy:.4f}"
    )

    print(
        f"Interpretation: "
        f"{interpretation}"
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "evaluation": {
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "interpretation": interpretation,
        },
    }