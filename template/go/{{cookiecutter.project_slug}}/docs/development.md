# Development Guide for {{cookiecutter.project_slug}}

This guide provides instructions for setting up your development environment, building the project, running tests, and other common development tasks.

## Prerequisites

Before you begin, ensure you have the following installed:

*   **Go:** Version `{{cookiecutter.go_version}}` or later. You can download it from [golang.org](https://golang.org/dl/).
*   **`make` utility:** Required to use the Makefile for common tasks.
*   **`pre-commit`:** For managing Git hooks. Install via `pip install pre-commit` or `uv pip install pre-commit`.
*   **`golangci-lint`:** (Optional, but recommended for `make lint`) The Go template's `make lint` uses this. Install it from [golangci-lint.run](https://golangci-lint.run/usage/install/). Pre-commit hooks might also install/run it.

## Setting Up the Environment

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/{{cookiecutter.__gh_slug}}.git
    cd {{cookiecutter.project_slug}}
    ```

2.  **Initialize the Project:**
    This command downloads necessary Go modules and sets up pre-commit Git hooks.
    ```bash
    make init
    ```

## Common Development Tasks (Using Makefile)

The Makefile provides several targets to streamline development:

*   **Format Code:**
    Formats your Go code using `go fmt`.
    ```bash
    make fmt
    ```

*   **Check Formatting (for CI):**
    Checks if the code is formatted correctly without making changes.
    ```bash
    make fmt-check
    ```

*   **Lint Code:**
    Runs `golangci-lint` to analyze your code for potential issues.
    ```bash
    make lint
    ```

*   **Build the Application:**
    Compiles the application. The binary will be placed in `./bin/{{cookiecutter.project_slug}}`.
    ```bash
    make build
    ```

*   **Run Tests:**
    Executes the test suite.
    ```bash
    make test
    ```

*   **Run the Application:**
    Builds (if necessary) and runs the compiled application. By default, it prints a welcome message and demonstrates config loading and a version subcommand.
    ```bash
    make run
    ```
    You can also run the compiled binary directly:
    ```bash
    ./bin/{{cookiecutter.project_slug}} [command] [flags]
    ```

*   **Clean Build Artifacts:**
    Removes the `./bin` directory and other build caches.
    ```bash
    make clean
    ```

*   **View All Makefile Targets:**
    Lists all available `make` commands with descriptions.
    ```bash
    make help
    ```

## Project Structure

A brief overview of the project's directory structure:

-   `cmd/`: Contains the main application code (using Cobra for CLI structure).
    - `root.go`: Defines the root command.
    - `version.go`: Defines the `version` subcommand.
-   `config/`: (Optional) Place for configuration files (e.g., `config.yaml`).
-   `docs/`: Contains detailed documentation.
-   `internal/`: (Optional) For private application and library code. Not meant for import by other projects.
-   `pkg/`: (Optional) For library code that's okay to be used by external applications.
-   `main.go`: The main entry point of the application.
-   `go.mod`, `go.sum`: Go module files.
-   `Makefile`: Defines common development tasks.
-   `README.md`: Overview and basic instructions.

## Contributing

Please refer to the main `CONTRIBUTING.md` file for guidelines on how to contribute to this project.
