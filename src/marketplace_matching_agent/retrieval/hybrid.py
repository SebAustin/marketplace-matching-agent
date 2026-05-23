from __future__ import annotations


async def bm25_search(query: str, k: int) -> list[tuple[str, float]]:
    raise NotImplementedError


async def dense_search(query: str, k: int, tower: str) -> list[tuple[str, float]]:
    raise NotImplementedError


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    raise NotImplementedError


async def cohere_rerank(
    query: str, candidates: list[str], top_n: int
) -> list[tuple[str, float]]:
    raise NotImplementedError


async def hybrid_retrieve(
    query: str, tower: str, k: int = 50, rerank_top_n: int = 10
) -> list[dict[str, object]]:
    raise NotImplementedError
