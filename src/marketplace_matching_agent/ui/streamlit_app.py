"""Streamlit demo dashboard stub."""

from __future__ import annotations

import asyncio
from typing import Literal, cast

import streamlit as st

from marketplace_matching_agent.graph import build_supervisor
from marketplace_matching_agent.state import MatchState

st.set_page_config(page_title="Marketplace Matching Agent", layout="wide")
st.title("marketplace-matching-agent")

tab_query, tab_fairness, tab_audit = st.tabs(["Query", "Fairness Slices", "Audit Trail"])

with tab_query:
    mode = st.radio("Mode", ["seeker", "recruiter"], horizontal=True)
    query = st.text_input("Query", "python backend austin")
    k = st.slider("Top k", 1, 20, 5)
    if st.button("Run"):
        graph = build_supervisor()
        result = asyncio.run(
            graph.ainvoke(
                cast(
                    MatchState,
                    {"mode": cast(Literal["seeker", "recruiter"], mode), "query": query, "k": k},
                )
            )
        )
        st.json(result, expanded=False)

with tab_fairness:
    st.info("Fairness slice charts populated from audit_log in production.")

with tab_audit:
    st.info("Audit trail paginated view wired to Postgres audit_log.")
