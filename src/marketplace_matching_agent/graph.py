"""LangGraph supervisor graph."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from marketplace_matching_agent.agents.evaluation import run_evaluation
from marketplace_matching_agent.agents.fairness import run_fairness
from marketplace_matching_agent.agents.search import run_search
from marketplace_matching_agent.state import MatchState, MatchStateUpdate


async def search_node(state: MatchState) -> MatchStateUpdate:
    """Run hybrid retrieval; tower is chosen inside search from mode."""
    return await run_search(state)


async def evaluation_node(state: MatchState) -> MatchStateUpdate:
    """Cite and rank top-k retrieved items."""
    return await run_evaluation(state)


async def fairness_node(state: MatchState) -> MatchStateUpdate:
    """Audit ranked list, rebalance once if needed, append audit row."""
    return await run_fairness(state)


def build_supervisor() -> CompiledStateGraph:
    """Build and compile the match supervisor graph.

    Returns:
        Compiled StateGraph with edges
        START -> search -> evaluation -> fairness -> END.
    """
    graph = StateGraph(MatchState)
    graph.add_node("search", search_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("fairness", fairness_node)
    graph.add_edge(START, "search")
    graph.add_edge("search", "evaluation")
    graph.add_edge("evaluation", "fairness")
    graph.add_edge("fairness", END)
    return graph.compile()
