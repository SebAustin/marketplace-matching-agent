"""Build synthetic balanced resumes from name lists."""

from __future__ import annotations

import json
from pathlib import Path

NAMES = Path("data/synthetic/bertrand_mullainathan_names.json")


def main() -> None:
    names = json.loads(NAMES.read_text())
    docs = []
    idx = 0
    for group, label in [("group_a", "A"), ("group_b", "B")]:
        for name in names[group]:
            docs.append(
                {
                    "id": f"candidates_syn_{idx:03d}",
                    "text": f"{name} Smith. Python backend engineer Austin.",
                    "meta": {"synthetic": True, "synthetic_group": label},
                }
            )
            idx += 1
    out = Path("tests/fixtures/candidates_docs.json")
    out.write_text(json.dumps(docs, indent=2))
    print(f"wrote {len(docs)} synthetic resumes to {out}")


if __name__ == "__main__":
    main()
