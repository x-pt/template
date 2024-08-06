# {{project-name}}

[![CI](https://github.com/{{gh_uname}}/{{project-name}}/workflows/CI/badge.svg)](https://github.com/{{gh_uname}}/{{project-name}}/actions)
[![Coverage Status](https://coveralls.io/repos/github/{{gh_uname}}/{{project-name}}/badge.svg?branch=main)](https://coveralls.io/github/{{gh_uname}}/{{project-name}}?branch=main)

## Overview

{{project-name}} is a Python project that aims to {{describe your project's purpose}}.

## Installation

### Using pip

To install `{{project-name}}`, simply use pip:

```bash
pip install {{project-name}}
```

### From source

```bash
# 1. Clone the repository:
git clone https://github.com/{{gh_uname}}/{{project-name}}.git

# 2. Navigate into the project directory:
cd {{project-name}}

# 3. Install the dependencies:
pip install -r requirements.txt

```

## Development

If you want to contribute to the project, follow these steps to set up the development environment:

Please ensure `uv`, `ruff` and `hatch` are installed firstly.

```bash
# create a virtual env by uv
make init

# install some dependencies by pyproject.toml with uv
make install-dev

# build wheel package by hatch
make build

# run the UT and FT by hatch
make test

```

## Usage

To use {{project-name}}, you can follow this.

```python
import {{project-name}}

# Example usage
{{project-name}}.do_something()
```

## License

Licensed under either of

* Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
* MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md).
