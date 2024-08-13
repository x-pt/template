# {{project-name}}

[![CI](https://github.com/{{cookiecutter.github_username}}/{{project-name}}/workflows/CI/badge.svg)](https://github.com/{{cookiecutter.github_username}}/{{project-name}}/actions)
[![Coverage Status](https://coveralls.io/repos/github/{{cookiecutter.github_username}}/{{project-name}}/badge.svg?branch=main)](https://coveralls.io/github/{{cookiecutter.github_username}}/{{project-name}}?branch=main)
[![PyPI version](https://badge.fury.io/py/{{project-name}}.svg)](https://badge.fury.io/py/{{project-name}})
[![Python Versions](https://img.shields.io/pypi/pyversions/{{project-name}}.svg)](https://pypi.org/project/{{project-name}}/)

## Overview

{{project-name}} is a Python project that aims to ...

## Features

- Feature 1: Description of feature 1
- Feature 2: Description of feature 2
- ...

## Requirements

- Python 3.11+
- Dependencies: list any major dependencies here

## Installation

```bash
pip install {{project-name}}
```

## Quick Start

Here's a simple example to get you started:

```python
import {{project-name}}

# Example usage
result = {{project-name}}.do_something()
print(result)
```

## Usage

For more detailed usage instructions, please refer to our [Usage Guide](docs/usage.md).

## Development

To set up the development environment:

1. Ensure `uv`, `ruff`, and `hatch` are installed.
2. Clone the repository and navigate to the project directory.
3. Run the following commands:

```bash
make init        # Create a virtual env using uv
make install-dev # Install development dependencies
make build       # Build wheel package
make test        # Run tests
```

For more details, see our [Contributing Guide](CONTRIBUTING.md).

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a history of changes to this project.
