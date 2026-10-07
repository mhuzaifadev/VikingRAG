.PHONY: install dev lint typecheck test test-unit test-integration test-eval migrate docker-up docker-down docker-reset pre-commit index

UV ?= uv

install:
	$(UV) sync --all-extras
	$(UV) run pre-commit install || true

dev:
	$(UV) run uvicorn vikingrag.api.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

typecheck:
	$(UV) run mypy src

test: test-unit

test-unit:
	$(UV) run pytest tests/unit tests/evaluation -q

test-integration:
	$(UV) run pytest tests/integration -q -m integration

test-eval:
	$(UV) run vikingrag-eval smoke --offline

migrate:
	$(UV) run vikingrag-migrate upgrade head

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

docker-reset:
	docker compose down -v

pre-commit:
	$(UV) run pre-commit run --all-files

# Index a document after structural ingest. Requires DOCUMENT_ID and a running API.
index:
	@test -n "$(DOCUMENT_ID)" || (echo "Usage: make index DOCUMENT_ID=<uuid>"; exit 1)
	curl -s -X POST "http://localhost:8000/v1/documents/$(DOCUMENT_ID)/index" \
	  -H "Content-Type: application/json" \
	  -d '{"force_summaries":false,"force_embeddings":false}' | jq
