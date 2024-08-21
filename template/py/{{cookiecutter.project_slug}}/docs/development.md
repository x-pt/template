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

- Python {{cookiecutter.python_version}}+

### Installation Steps

1. Install `uv`

    ```bash
    # macOS and Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Windows
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

2. Clone the repository and navigate to the project directory:

    ```bash
    git clone https://github.com/{{cookiecutter.__gh_slug}}.git
    cd {{cookiecutter.project_slug}}
    ```

3. Set up the development environment:

    ```bash
    make init
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
