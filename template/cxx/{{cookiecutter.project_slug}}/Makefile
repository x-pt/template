.PHONY: help init build run test lint-check clean image
.DEFAULT_GOAL := help

APP_NAME := {{cookiecutter.project_slug}}

# init
init:
	@pre-commit install --hook-type commit-msg --hook-type pre-push

# compile and build
build:
{%- if cookiecutter.cxx_build_tool == "cmake" %}
	@cmake -S . -B build
	@cmake --build build --parallel
{%- else %}
	@xmake -y
{%- endif %}

# run
run:
{%- if cookiecutter.cxx_build_tool == "cmake" %}
	@echo "Running $(APP_NAME) (CMake build)..."
	@./build/bin/$(APP_NAME) $(ARGS)
{%- else %}
	@echo "Running $(APP_NAME) (XMake build)..."
	@xmake run $(APP_NAME) $(ARGS)
{%- endif %}

# Run tests
test: build # Ensure project is built before testing
{%- if cookiecutter.cxx_build_tool == "cmake" %}
	@echo "Running C++ tests (GoogleTest executable via CMake build)..."
	@./build/bin/{{cookiecutter.project_slug}}-tests $(ARGS)
{%- else %}
	@echo "Running C++ tests (GoogleTest executable via XMake build)..."
	@xmake run {{cookiecutter.project_slug}}-tests $(ARGS)
{%- endif %}

# Run pre-commit checks on all files
lint-check:
	@echo "Running pre-commit checks..."
	@pre-commit run --all-files

# Clean build artifacts
clean:
{%- if cookiecutter.cxx_build_tool == "cmake" %}
	@echo "Cleaning CMake build directory..."
	@rm -rf build
{%- else %}
	@echo "Cleaning XMake artifacts..."
	@xmake clean
{%- endif %}
	@echo "Cleaning Docker images..."
	@docker compose -f ./compose.yml down -v --rmi local 2>/dev/null || true
	@docker image rm -f $(APP_NAME) 2>/dev/null || true

# build image
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
