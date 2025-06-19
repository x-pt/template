.PHONY: help init sync build run test fmt fmt-check lint allure image compose-up compose-down clean
.DEFAULT_GOAL := help

APP_NAME := {{cookiecutter.project_slug}}

# Initialize the project, install dependencies and pre-commit hooks
init: sync
	@echo "Installing pre-commit hooks..."
	@uv run pre-commit install --hook-type commit-msg --hook-type pre-push

# Synchronize the virtual environment with pyproject.toml
sync:
	@echo "Syncing Python environment with uv..."
	@uv sync --dev

# Build wheel and source distribution
build:
	@uv build

# Run the Python application (if defined as a script)
run:
	@uv run {{cookiecutter.project_slug}}

# Run Pytest tests
test:
	@uv run pytest

# Format Python source code using Ruff
fmt:
	@uv run ruff format .

# Check Python code formatting using Ruff (for CI)
fmt-check:
	@echo "Checking Python code formatting with Ruff..."
	@uv run ruff format --check .

# Lint Python source code using Ruff and MyPy
lint:
	@uv run ruff check . --fix
	@uv run mypy .

# Generate and serve Allure test report
allure:
	@allure serve allure-results

# Build Docker image
image:
	@docker image build -t $(APP_NAME) .

# Start services defined in compose.yml
compose-up:
	@docker compose -f ./compose.yml -p $(APP_NAME) up -d

# Stop and remove services defined in compose.yml
compose-down:
	@docker compose -f ./compose.yml down

# Clean build artifacts, test reports, and Docker assets
clean:
	@rm -rf build dist *.egg-info htmlcov .coverage coverage.xml coverage.lcov
	@docker compose -f ./compose.yml down -v
	@docker image rm -f $(APP_NAME)

# Show help
help:
	@echo ""
	@echo "Usage:"
	@echo "    make [target]"
	@echo ""
	@echo "Targets:"
	@awk '/^[a-zA-Z\-_0-9]+:/ \
	{ \
		helpMessage = match(lastLine, /^# (.*)/); \
		if (helpMessage) { \
			helpCommand = substr($$1, 0, index($$1, ":")-1); \
			helpMessage = substr(lastLine, RSTART + 2, RLENGTH); \
			printf "\033[36m%-22s\033[0m %s\n", helpCommand,helpMessage; \
		} \
	} { lastLine = $$0 }' $(MAKEFILE_LIST)
