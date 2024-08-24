.PHONY: help build image
.DEFAULT_GOAL := help

APP_NAME := {{cookiecutter.project_slug}}

# compile and build
build:
{%- if cookiecutter.cxx_build_tool == "cmake" %}
	@cmake -S . -B build
	@cmake --build build
{%- else %}
	@xmake
{%- endif %}

{% if cookiecutter.cxx_project_type == "binary" -%}
# run
run:
	@./build/bin/{{cookiecutter.project_slug}}
{%- endif %}

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
