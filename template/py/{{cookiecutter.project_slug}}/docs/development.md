# Development Guide for {{cookiecutter.project_slug}}

Welcome to the development guide for `{{cookiecutter.project_slug}}`!
This document will walk you through setting up your development environment, running tests, building the project, and maintaining code quality.

## Table of Contents

- [Setting Up the Development Environment](#setting-up-the-development-environment)
    - [Prerequisites](#prerequisites)
    - [Installation Steps](#installation-steps)
- [Running Tests](#running-tests)
- [Building the Project](#building-the-project)
- [Code Style and Linting](#code-style-and-linting)

## Setting Up the Development Environment

### Prerequisites

Before you start, make sure you have the following installed on your system:

- **Python {{cookiecutter.python_version}}+**: Ensure you have the correct version of Python. You can check your Python version with:

    ```bash
    python --version
    ```

- **`uv` tool**: This tool helps manage your Python environment.

    - **macOS and Linux**:

        ```bash
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```

    - **Windows**:

        ```bash
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

### Installation Steps

1. **Clone the Repository**: Start by cloning the project repository to your local machine and navigate to the project directory:

    ```bash
    git clone https://github.com/{{cookiecutter.__gh_slug}}.git
    cd {{cookiecutter.project_slug}}
    ```

2. **Initialize the Development Environment**: This command sets up a virtual environment and installs all necessary dependencies.

    ```bash
    make init
    ```

    This step will also install any pre-commit hooks, ensuring your code adheres to the project’s coding standards before each commit.

## Running Tests

Tests are crucial to ensure the stability of the project. To run all tests, use the following command:

```bash
make test
```

This command will execute the test suite using `pytest`, ensuring all components work as expected.

[Consider adding specific details on the structure of tests, testing strategy, or how to add new tests.]

## Building the Project

To build the project and create a distributable package, use:

```bash
make build
```

This command will generate a `.whl` file in the `dist` directory, which can be used to distribute and install the project.

## Code Style and Linting

Maintaining consistent code style is essential. We use `ruff` for linting and formatting. To check for any style issues, run:

```bash
make ruff
```

This command will automatically check and optionally fix any code style issues according to the project's style guide.

---

By following this guide, you'll be well on your way to contributing to `{{cookiecutter.project_slug}}`. Thank you for your efforts in maintaining and improving this project!
