.PHONY: help init build run test fmt fmt-check lint clean image
.DEFAULT_GOAL := help

APP_NAME := {{cookiecutter.project_slug}}
BINARY_NAME := {{cookiecutter.project_slug}}
CMD_PATH := .

# Initialize the project, install dependencies and pre-commit hooks
init:
	@echo "Tidying and downloading Go modules..."
	@go mod tidy
	@go mod download
	@echo "Installing pre-commit hooks..."
	@pre-commit install --hook-type commit-msg --hook-type pre-push

# Build the Go application
build:
	@go build -trimpath -ldflags="-w -s" -o ./bin/$(BINARY_NAME) $(CMD_PATH)

# Run the Go application (after building)
run:
	@$(MAKE) build && ./bin/$(BINARY_NAME)

# Run Go tests with coverage
test:
	@go test -v -cover ./...

# Format Go source code
fmt:
	@go fmt ./...

# Check Go code formatting (for CI)
fmt-check:
	@echo "Checking Go code formatting..."
	@test -z "$$(gofmt -l .)" || (echo "Go code is not formatted. Please run 'make fmt'."; false)

# Lint Go source code
lint:
	@golangci-lint run

# Clean build artifacts and Docker images
clean:
	@echo "Cleaning Go build artifacts..."
	@rm -rf ./bin
	@go clean -cache
	@echo "Cleaning Docker images..."
	@docker compose -f ./compose.yml down -v --rmi local 2>/dev/null || true
	@docker image rm -f $(APP_NAME) 2>/dev/null || true

# Build Docker image
image:
	@docker image build -t $(APP_NAME) .

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
