"""Compute eval delta vs main branch summary."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: eval_delta.py <pr_summary.json> <baseline_branch>")
        sys.exit(1)
    pr_path = Path(sys.argv[1])
    pr = json.loads(pr_path.read_text())
    lines = [
        "## Eval smoke",
        f"- nDCG@10: {pr.get('ndcg_at_10', 0):.3f}",
        f"- MRR: {pr.get('mrr', 0):.3f}",
        f"- GoodMatch@k: {pr.get('goodmatch_at_k', 0):.3f}",
        f"- 4/5ths pass rate: {pr.get('four_fifths_pass_rate', 0):.3f}",
        f"- p95 latency ms: {pr.get('p95_latency_ms', 0):.1f}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
