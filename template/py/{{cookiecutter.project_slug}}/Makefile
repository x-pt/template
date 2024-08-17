.PHONY: help init install-dev build test ruff image clean
.DEFAULT_GOAL := help

APP_NAME := {{cookiecutter.project_slug}}

# Create virtual environment
init:
	@uv venv

# Install development dependencies
install-dev:
	@uv pip install .[dev]

# Build wheel
build:
	@hatch build

# Test
test:
	@hatch test

# Ruff
ruff:
	@ruff format .
	@ruff check . --fix

# Build image
image:
	@docker image build -t $(APP_NAME) .

# Clean build artifacts
clean:
	@rm -rf build dist *.egg-info htmlcov .coverage coverage.xml coverage.lcov

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
