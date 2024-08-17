# {{cookiecutter.project_slug}}

[![CI](https://github.com/{{cookiecutter.__gh_slug}}/workflows/CI/badge.svg)](https://github.com/{{cookiecutter.__gh_slug}}/actions)
[![Coverage Status](https://coveralls.io/repos/github/{{cookiecutter.__gh_slug}}/badge.svg?branch=main)](https://coveralls.io/github/{{cookiecutter.__gh_slug}}?branch=main)
[![PyPI version](https://badge.fury.io/py/{{cookiecutter.project_slug}}.svg)](https://badge.fury.io/py/{{cookiecutter.project_slug}})
[![Python Versions](https://img.shields.io/pypi/pyversions/{{cookiecutter.project_slug}}.svg)](https://pypi.org/project/{{cookiecutter.project_slug}}/)

## Overview

{{cookiecutter.project_slug}} is a Python project that aims to ...

## Features

- Feature 1: Description of feature 1
- Feature 2: Description of feature 2
- ...

## Requirements

- Python 3.11+
- Dependencies: list any major dependencies here

## Installation

```bash
pip install {{cookiecutter.project_slug}}
```

## Quick Start

Here's a simple example to get you started:

```python
import {{cookiecutter.project_slug}}

# Example usage
result = {{cookiecutter.project_slug}}.do_something()
print(result)
```

## Usage

For more detailed usage instructions, please refer to our [Usage Guide](docs/usage.md).

## Development

To set up the development environment:

1. Ensure `uv`, `ruff`, and `hatch` are installed.

    - Using `pipx`
        ```bash
        # you can use `pipx`
        pipx install uv ruff hatch

        # or
        # Firstly, install `uv` (macOS and Linux)
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # (Windows)
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

        # Then
        uv tool install ruff
        uv tool install hatch
        ```
    - Using `uv`
        ```bash
        # macOS and Linux
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Windows
        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

        # Then
        uv tool install ruff
        uv tool install hatch
        ```
    - BTW
        > `pipx` and `uv tool install` will usually install the binaries in your `~/.local/bin` directory.
        >
        > Please ensure the `~/.local/bin` directory is in your `PATH` environment variable.

2. Clone the repository and navigate to the project directory.
3. Run the following commands:

```bash
make init         # Create a virtual env using uv
make install-dev  # Install development dependencies
make build        # Build wheel package
make test         # Run tests
make image        # Build Docker image
make compose-up   # Run Docker Compose
make compose-down # Stop and remove Docker Compose
make clean        # Remove build artifacts
```

For more details, see our [Contributing Guide](CONTRIBUTING.md).

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of changes to this project.
