# GridSmart -- common tasks.
.PHONY: help install test lint fmt up down producer clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and dev dependencies
	pip install -e ".[dev]"

test:  ## Run the test suite
	pytest tests -q --cov=gridsmart --cov-report=term-missing

lint:  ## Check PEP 8, import order and formatting
	ruff check src tests
	black --check src tests

fmt:  ## Auto-format
	ruff check --fix src tests
	black src tests

up:  ## Start Kafka + JupyterLab (http://localhost:8888)
	docker compose up -d
	@echo "JupyterLab: http://localhost:8888   Spark UI: http://localhost:4040"

down:  ## Stop the stack and remove volumes
	docker compose down -v

producer:  ## Run the Kafka producer directly (outside the notebook)
	python -m gridsmart.producer

clean:  ## Remove streaming state, checkpoints and caches
	rm -rf streamoutput/ spark-warehouse/ metastore_db/ derby.log
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
