# Development Guide for {{cookiecutter.project_slug}}

Welcome to the development guide for `{{cookiecutter.project_slug}}`!
This document will walk you through setting up your development environment, running tests, building the project, and maintaining code quality using the provided `Makefile`.

## Table of Contents

- [Setting Up the Development Environment](#setting-up-the-development-environment)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
- [Code Formatting, Linting, and Type Checking](#code-formatting-linting-and-type-checking)
- [Running Tests](#running-tests)
- [Building the Project](#building-the-project)
- [Other Development Commands](#other-development-commands)
- [Running the Application](#running-the-application)


## Setting Up the Development Environment

### Prerequisites

Before you start, make sure you have the following installed on your system:

-   **Python `{{cookiecutter.python_version}}`+**: Ensure you have the correct version of Python. You can check your Python version with `python --version`.
-   **`uv` tool**: For Python environment and package management. Installation instructions:
    -   **macOS and Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
    -   **Windows**: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
    More details can be found on the [official `uv` installation page](https://astral.sh/uv#installation).
-   **`make` utility**: For running development commands defined in the Makefile. This is typically pre-installed on Linux and macOS. Windows users might need to install it (e.g., via Chocolatey, MSYS2, or WSL).
-   **`pre-commit`**: For managing Git hooks. Install with `pip install pre-commit` or `uv pip install pre-commit` (note: `make init` will install it into the project's virtual environment if not globally available and `uv` is used).

### Installation Steps

1.  **Clone the Repository**: Start by cloning the project repository to your local machine and navigate to the project directory:

    ```bash
    git clone https://github.com/{{cookiecutter.__gh_slug}}.git
    cd {{cookiecutter.project_slug}}
    ```

2.  **Initialize the Development Environment**: This command is crucial. It sets up a Python virtual environment using `uv`, installs all necessary dependencies (including development dependencies from `pyproject.toml`), and sets up pre-commit Git hooks.

    ```bash
    make init
    ```
    This step ensures your environment is consistent and that code quality checks are run automatically before each commit.

## Code Formatting, Linting, and Type Checking

To maintain code quality and consistency, we use Ruff for formatting and linting, and MyPy for static type checking. These tools are managed via `make` targets and pre-commit hooks.

-   **Format code:** Automatically formats your code using Ruff according to the rules defined in `pyproject.toml`.
    ```bash
    make fmt
    ```
-   **Check formatting (for CI):** Checks if the code is formatted correctly without making changes. This is useful for CI pipelines.
    ```bash
    make fmt-check
    ```
-   **Lint and type-check code:** Runs Ruff for linting (with auto-fix enabled for safe fixes) and MyPy for static type checking.
    ```bash
    make lint
    ```
These checks are also enforced by pre-commit hooks installed via `make init`, which run on staged files before you commit.

## Running Tests

Tests are crucial to ensure the stability and correctness of the project. To run all tests (unit tests, integration tests, etc.) and generate a coverage report:

```bash
make test
```
This command will execute the test suite using `pytest`. Test configurations can be found in `pyproject.toml` under the `[tool.pytest.ini_options]` section.

## Building the Project

To build the project into distributable packages (wheel and source distribution):

```bash
make build
```
This command uses `uv build` and will generate the packages in the `dist/` directory. These packages can then be uploaded to PyPI or other package repositories.

## Running the Application

This template includes a default runnable application which is a simple HTTP server. To run it:

```bash
make run
```
This will start the server on `http://localhost:8000` by default, serving files from the current directory. Press `Ctrl+C` to stop it. The application logic is in `src/{{cookiecutter.package_name}}/app.py`.

## Other Development Commands

The `Makefile` provides several other useful targets:

-   **Generate and serve Allure test report:**
    If you have Allure installed and your tests generate Allure results (e.g., via `pytest --alluredir=allure-results`), you can serve the report locally.
    ```bash
    make allure
    ```
-   **Manage Docker Compose services (if defined in `compose.yml`):**
    ```bash
    make compose-up  # Start services in detached mode
    make compose-down # Stop and remove services
    ```
-   **Build Docker image:**
    Builds a Docker image for the application as defined in the `Dockerfile`.
    ```bash
    make image
    ```
-   **Clean build artifacts and virtual environment:**
    Removes build directories (e.g., `build/`, `dist/`), Python caches (e.g., `*.egg-info`, `__pycache__`), test artifacts (`htmlcov/`, `.coverage`, `allure-results/`), and the `uv` virtual environment (`.venv/`).
    ```bash
    make clean
    ```
-   **View all available `make` targets:**
    Displays a help message listing all defined targets and their descriptions.
    ```bash
    make help
    ```

---

By following this guide, you'll be well on your way to contributing to `{{cookiecutter.project_slug}}`. Thank you for your efforts in maintaining and improving this project!
