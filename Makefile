.PHONY: install dev lint typecheck test test-unit test-integration migrate docker-up docker-down pre-commit

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
	$(UV) run pytest tests/unit -q

test-integration:
	$(UV) run pytest tests/integration -q -m integration

migrate:
	$(UV) run alembic upgrade head

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down -v

pre-commit:
	$(UV) run pre-commit run --all-files
