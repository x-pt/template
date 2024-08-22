# {{cookiecutter.project_slug}}

[![CI](https://github.com/{{cookiecutter.__gh_slug}}/workflows/CI/badge.svg)](https://github.com/{{cookiecutter.__gh_slug}}/actions)
[![Coverage Status](https://coveralls.io/repos/github/{{cookiecutter.__gh_slug}}/badge.svg?branch=main)](https://coveralls.io/github/{{cookiecutter.__gh_slug}}?branch=main)

## Installation

### Build

- Ensure you have a C++ compiler installed (e.g., `g++`, `clang++`).
- Install [CMake](https://cmake.org/install/) and any necessary dependencies.
- Clone the repository:
    ```sh
    git clone https://github.com/{{cookiecutter.__gh_slug}}.git
    cd {{cookiecutter.project_slug}}
    make build
    ./{{cookiecutter.project_slug}}
    ```

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

See [CONTRIBUTING.md](CONTRIBUTING.md).
