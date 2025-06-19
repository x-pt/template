.PHONY: help init build run test fmt fmt-check lint doc image compose-up compose-down clean
.DEFAULT_GOAL := help

APP_NAME := {{cookiecutter.project_slug}}

# Initialize the project, fetch dependencies, and install pre-commit hooks
init:
	@echo "Fetching Rust dependencies..."
	@cargo fetch
	@echo "Installing pre-commit hooks..."
	@pre-commit install --hook-type commit-msg --hook-type pre-push

# Build the Rust project in release mode
build:
	@cargo build --release

# Run the Rust application (release build)
run:
	@cargo run --release

# Run Rust tests
test:
	@cargo test

# Format Rust source code using cargo fmt
fmt:
	@cargo fmt

# Check Rust code formatting (for CI)
fmt-check:
	@echo "Checking Rust code formatting..."
	@cargo fmt --all --check

# Lint Rust source code using Clippy
lint:
	@cargo clippy --all-targets --all-features -- -D warnings

# Generate Rust documentation
doc:
	@cargo doc --no-deps

# Build image
image:
	@docker image build -t $(APP_NAME) .

# Start a compose service
compose-up:
	@docker compose -f ./compose.yml -p $(APP_NAME) up -d

# Shutdown a compose service
compose-down:
	@docker compose -f ./compose.yml down

# Clean build artifacts and Docker assets
clean:
	@echo "Cleaning Rust build artifacts..."
	@cargo clean
	@echo "Cleaning Docker images..."
	@docker compose -f ./compose.yml down -v --rmi local 2>/dev/null || true
	@docker image rm -f $(APP_NAME) 2>/dev/null || true

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
