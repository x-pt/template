.PHONY: help init build run test fmt fmt-check lint image compose-up compose-down clean
.DEFAULT_GOAL := help

APP_NAME := {{cookiecutter.project_slug}}

# Initialize the project and install dependencies
init:
	@pnpm install

# Build the TypeScript application
build:
	@pnpm build

# Run the built TypeScript application
run:
	@$(MAKE) build && node dist/index.js

# Run tests using the configured test runner (Jest)
test:
	@pnpm test

# Format source code using Biome
fmt:
	@pnpm biome format --write .

# Check TypeScript code formatting and linting using Biome (for CI)
fmt-check:
	@echo "Checking TypeScript code with Biome..."
	@pnpm exec biome check .

# Lint source code using Biome (via pnpm script)
lint:
	@pnpm lint

# Build Docker image
image:
	@docker image build -t $(APP_NAME) .

# Start services defined in compose.yml
compose-up:
	@docker compose -f ./compose.yml -p $(APP_NAME) up -d

# Stop and remove services defined in compose.yml
compose-down:
	@docker compose -f ./compose.yml down

# Clean build artifacts, dependencies, and Docker assets
clean:
	@echo "Cleaning TypeScript build artifacts and dependencies..."
	@rm -rf node_modules dist lib coverage .nyc_output
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
