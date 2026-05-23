import pytest

from marketplace_matching_agent.graph import (
    build_supervisor,
    evaluation_node,
    fairness_node,
    search_node,
)


@pytest.mark.asyncio
async def test_search_node_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await search_node({"mode": "seeker", "query": "python", "k": 5})


@pytest.mark.asyncio
async def test_evaluation_node_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await evaluation_node({"mode": "seeker", "query": "python", "k": 5})


@pytest.mark.asyncio
async def test_fairness_node_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        await fairness_node({"mode": "seeker", "query": "python", "k": 5})


def test_build_supervisor_compiles() -> None:
    graph = build_supervisor()
    assert graph is not None
