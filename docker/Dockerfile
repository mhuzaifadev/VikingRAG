FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./

RUN uv pip install --system -e ".[dev]"

EXPOSE 8000

CMD ["uvicorn", "vikingrag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
