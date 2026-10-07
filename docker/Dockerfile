FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv \
    && groupadd --gid 10001 vikingrag \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin vikingrag \
    && mkdir -p /app/data/object_store \
    && chown -R vikingrag:vikingrag /app/data

COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./

# Runtime image: production deps only. Compose prod sets user 10001:10001.
RUN uv pip install --system -e "." \
    && chown -R vikingrag:vikingrag /app

EXPOSE 8000

CMD ["uvicorn", "vikingrag.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
