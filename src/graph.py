from langgraph.graph import StateGraph, START, END
from src.state import AgentState
from src.agents.retriever_agent import retriever_agent
from src.agents.response_agent import response_agent
from src.agents.evaluator_agent import evaluator_agent

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("response", response_agent)
    graph.add_node("evaluator", evaluator_agent)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "response")
    graph.add_edge("response", "evaluator")
    graph.add_edge("evaluator", END)
    return graph.compile()
