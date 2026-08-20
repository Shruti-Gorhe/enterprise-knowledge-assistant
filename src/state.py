from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict, total=False):
    question: str
    retrieved_context: List[str]
    answer: str
    sources: List[Dict[str, Any]]
    faithfulness: float
    answer_relevancy: float
    evaluation: Dict[str, Any]
