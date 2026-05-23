"""Shared type aliases for marketplace-matching-agent."""

from __future__ import annotations

from marketplace_matching_agent.state import FairnessReport, Rationale

type ItemDict = dict[str, object]
type RankedList = tuple[list[ItemDict], FairnessReport, list[Rationale]]
