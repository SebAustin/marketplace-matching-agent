from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from marketplace_matching_agent.state import MatchState


async def search_node(state: MatchState) -> MatchState:
    raise NotImplementedError


async def evaluation_node(state: MatchState) -> MatchState:
    raise NotImplementedError


async def fairness_node(state: MatchState) -> MatchState:
    raise NotImplementedError


def build_supervisor() -> CompiledStateGraph[Any, Any, Any, Any]:
    g = StateGraph(MatchState)
    g.add_node("search", search_node)
    g.add_node("evaluation", evaluation_node)
    g.add_node("fairness", fairness_node)
    g.add_edge(START, "search")
    g.add_edge("search", "evaluation")
    g.add_edge("evaluation", "fairness")
    g.add_edge("fairness", END)
    return g.compile()
