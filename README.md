[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

# X Project Template

A [**CookieCutter**](https://github.com/cookiecutter/cookiecutter)-based template designed to serve as a robust starting point for most startup projects hosted on GitHub.

This template aims to streamline the initial setup and provide a solid foundation for new projects, ensuring best practices are followed from the start.

Visit our [website](https://x-pt.github.io) for more details.

## Features

- **Clean and Modular Structure**: Predefined directory structure to keep your code organized.
- **CI/CD Integration**: Pre-configured GitHub Actions workflows for automated testing and deployment.
- **Pre-configured Linters**: Integrated with some tools to maintain code quality.
- **Docker Support**: Ready-to-use Dockerfile and Docker Compose setup for easy containerization.
- **Customizable Templates**: Modify to fit your project’s specific needs with minimal effort.

## Getting Started

### Prerequisites

- Python 3.8+
- [CookieCutter](https://cookiecutter.readthedocs.io/en/latest/installation.html)

### Available Templates

This repository contains 6 sub-templates. After running `cookiecutter`, you will be prompted to select one:

| Template | Description | Key Options |
|----------|-------------|-------------|
| `cuda` | CUDA project with CMake/XMake | `cxx_build_tool`, `cuda_standard_version` |
| `cxx` | C++ project with CMake/XMake | `cxx_build_tool`, `cxx_standard_version` |
| `go` | Go project | `go_version` |
| `py` | Python project | `python_version` |
| `rs` | Rust project | `crate_type` (bin/lib) |
| `ts` | TypeScript project | `node_version` |

### Usage

To generate a new project, run `cookiecutter` and select a template from the prompt:

```bash
cookiecutter gh:x-pt/template
```

You will be prompted to:
1. **Select a template** (cuda, cxx, go, py, rs, ts)
2. **Enter project details** (name, slug, description, version, author info, etc.)
3. **Configure template-specific options** (e.g., `cxx_build_tool`, `crate_type`, etc.)

Example selecting the Python template non-interactively:

```bash
cookiecutter --no-input -o my-project gh:x-pt/template templates:py
```

### Template-Specific Details

**C++ / CUDA** — Choose between CMake and XMake as the build system.
The selected build tool's files are kept; the other is removed.

**Rust** — Choose between `bin` (executable) and `lib` (library) crate type.
The unused entry point (`main.rs` or `lib.rs`) is removed.

**Python** — Uses uv for dependency management and project configuration.

**Go** — Standard Go module layout with `cmd/`, `internal/`, and `docs/`.

**TypeScript** — Uses npm with Husky for git hooks.

## Example Project

Check out an example project generated with this template:

- [Example for C++ with CMake](https://github.com/x-pt/example-cxx-cmake)
- [Example for C++ with XMake](https://github.com/x-pt/example-cxx-xmake)
- [Example for CUDA with CMake](https://github.com/x-pt/example-cuda-cmake)
- [Example for CUDA with XMake](https://github.com/x-pt/example-cuda-xmake)
- [Example for Golang](https://github.com/x-pt/example-go)
- [Example for Python](https://github.com/x-pt/example-py)
- [Example for Rust](https://github.com/x-pt/example-rs)
- [Example for TypeScript](https://github.com/x-pt/example-ts)

## For Template Maintainers

The 6 sub-templates share common cookiecutter variables (project name, slug,
author, etc.) defined in a single source of truth. To regenerate all
`cookiecutter.json` files after modifying the common definition or adding a new
template:

```bash
make gen-config
```

This runs `scripts/generate_cookiecutter_json.py`, which merges the common
variable definitions in the script with per-template overrides and writes
`cookiecutter.json` for each sub-template.

All hooks across sub-templates follow a consistent pattern: template-specific
cleanup (e.g., removing unused build files) followed by `git init` with
standardized error handling.

## Contributing

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.
