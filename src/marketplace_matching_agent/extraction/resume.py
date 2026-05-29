"""Resume parsing helpers."""

from __future__ import annotations

import re


def parse_resume_text(text: str) -> dict[str, object]:
    """Parse resume plain text into structured sections."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {"sections": lines[:10], "word_count": len(text.split())}


def extract_skills_from_text(text: str) -> dict[str, object]:
    """Extract capitalized skill-like tokens from resume text."""
    tokens = set(re.findall(r"\b[A-Z][a-zA-Z+#.]+\b", text))
    return {"skills": sorted(tokens)[:20]}
