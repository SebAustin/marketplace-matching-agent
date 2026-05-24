FROM python:3.12.4-slim

WORKDIR /app

RUN pip install --no-cache-dir uv==0.5.11

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY mcp_servers ./mcp_servers

RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "marketplace_matching_agent.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
