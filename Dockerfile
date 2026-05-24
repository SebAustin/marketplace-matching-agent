FROM ghcr.io/astral-sh/uv:0.5.11-python3.12-bookworm-slim AS builder

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY mcp_servers ./mcp_servers

RUN uv sync --frozen --no-dev

FROM python:3.12.4-slim-bookworm

RUN apt-get update \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade \
        "setuptools>=78.1.1" \
        "wheel>=0.46.2" \
        "jaraco.context>=6.1.0" \
    && pip uninstall -y pip \
    && rm -rf /root/.cache

WORKDIR /app

COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "marketplace_matching_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
