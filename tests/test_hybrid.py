import pytest

from marketplace_matching_agent.retrieval import hybrid


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fn,args",
    [
        (hybrid.bm25_search, ("python", 5)),
        (hybrid.dense_search, ("python", 5, "jobs")),
        (hybrid.cohere_rerank, ("python", ["doc"], 1)),
        (hybrid.hybrid_retrieve, ("python", "jobs")),
    ],
)
async def test_async_stubs_raise(fn, args) -> None:
    with pytest.raises(NotImplementedError):
        await fn(*args)


def test_rrf_fuse_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        hybrid.rrf_fuse([[]])
