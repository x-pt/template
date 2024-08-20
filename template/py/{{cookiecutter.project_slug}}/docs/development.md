# Development Guide for {{cookiecutter.project_slug}}

This guide will help you set up your development environment and contribute to the `{{cookiecutter.project_slug}}` project.

## Table of Contents
- [Setting Up the Development Environment](#setting-up-the-development-environment)
- [Running Tests](#running-tests)
- [Building the Project](#building-the-project)
- [Code Style and Linting](#code-style-and-linting)

## Setting Up the Development Environment

### Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.11+
- `pipx` (recommended for tool installation)

### Installation Steps

1. Install required tools:

    We recommend using `pipx` for installing `uv`, `ruff`, and `hatch`. If you don't have `pipx`, install it first:

    ```bash
    python -m pip install --user pipx
    python -m pipx ensurepath
    ```

    Then install the required tools:

    ```bash
    pipx install uv ruff hatch
    ```

    Alternative installation method using `uv`:
    - For macOS and Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv tool install ruff hatch
    ```

    - For Windows:
    ```powershell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    uv tool install ruff hatch
    ```

    > Note: Ensure that `~/.local/bin` is in your `PATH` environment variable.

2. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/{{cookiecutter.__gh_slug}}.git
    cd {{cookiecutter.project_slug}}
    ```

3. Set up the development environment:

    ```bash
    make init         # Create a virtual environment using uv
    make install-dev  # Install development dependencies
    ```

## Running Tests

To run the test suite:

```bash
make test
```

[Add more details about the test suite, how to write tests, etc.]

## Building the Project

To build the project:

```bash
make build
```

This will create a wheel package in the `dist` directory.

## Code Style and Linting

We use `ruff` for linting and code formatting. To run the linter:

```bash
make ruff
```

Thank you for contributing to `{{cookiecutter.project_slug}}`!