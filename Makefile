.PHONY: setup sync build test lint format clean

setup: ## Install uv (if needed) and sync workspace
	@command -v uv >/dev/null 2>&1 || { echo "Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv sync
	@echo "Installing git pre-commit hook..."
	@printf '#!/bin/sh\nSTAGED=$$(git diff --cached --name-only --diff-filter=ACMR)\nmake format\necho "$$STAGED" | xargs -r git add\n' > .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit

sync: ## Sync all workspace dependencies
	uv sync

build: ## Build all packages
	uv build --package flo-pro-sdk
	uv build --package flo-pro-adk

test: sync ## Run tests for all packages
	uv run pytest python/flo-pro-sdk/tests
	uv run pytest python/flo-pro-adk/tests --no-header -q 2>/dev/null || true

test-all: ## Run tests with all optional dependencies
	uv sync --extra all
	uv run pytest python/flo-pro-sdk/tests
	uv run pytest python/flo-pro-adk/tests --no-header -q 2>/dev/null || true

lint: ## Lint all code
	uv run ruff check python/
	uv run ruff format --check python/

format: ## Format all code
	uv run ruff check --fix python/
	uv run ruff format python/

clean: ## Remove build artifacts
	rm -rf python/flo-pro-sdk/dist
	rm -rf python/flo-pro-adk/dist
	rm -rf .venv

docs: ## Build documentation
	cd docs && uv run mkdocs build

docs-serve: ## Serve documentation locally
	cd docs && uv run mkdocs serve

docs-deploy: ## Deploy documentation to GitHub Pages
	cd docs && uv run mkdocs gh-deploy

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
