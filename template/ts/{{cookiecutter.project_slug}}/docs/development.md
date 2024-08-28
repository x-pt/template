# Development Guide for {{cookiecutter.project_slug}}

Welcome to the development guide for `{{cookiecutter.project_slug}}`! This document will guide you through setting up your development environment, running tests, building the project, and maintaining code quality.

## Table of Contents

- [Setting Up the Development Environment](#setting-up-the-development-environment)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [Running Tests](#running-tests)
- [Building the Project](#building-the-project)
- [Code Style and Linting](#code-style-and-linting)

## Setting Up the Development Environment

### Prerequisites

Before you begin, make sure you have the following installed on your system:

- **Node.js {{cookiecutter.node_version}}+**: Ensure you have the correct version of Node.js. You can check your Node.js version with:

  ```bash
  node --version
  ```

- **npm or yarn**: A package manager is required to install dependencies. You can check if you have npm or yarn installed with:

  ```bash
  npm --version
  yarn --version
  ```

### Installation Steps

1. **Clone the Repository**: Start by cloning the project repository to your local machine and navigate to the project directory:

    ```bash
    git clone https://github.com/{{cookiecutter.__gh_slug}}.git
    cd {{cookiecutter.project_slug}}
    ```

2. **Install Dependencies**: Use npm or yarn to install all necessary dependencies:

    ```bash
    npm install
    # or
    yarn install
    ```

    This step will also set up any pre-commit hooks defined in the project, ensuring your code adheres to the project’s coding standards.

## Running Tests

Running tests is crucial to ensure the stability of the project. To run all tests, use the following command:

```bash
npm test
# or
yarn test
```

This command will execute the test suite using `jest`, ensuring all components work as expected.

[You may include additional details on the testing framework, testing strategy, or how to add new tests.]

## Building the Project

To build the project and generate the compiled JavaScript files, use:

```bash
npm run build
# or
yarn build
```

This command will compile the TypeScript files into JavaScript and place them in the `dist` directory.

## Code Style and Linting

Maintaining consistent code style is essential. We use `biome` for linting and formatting. To check for any style issues and automatically fix them, run:

```bash
npm run lint
# or
yarn lint
```

This command will check the codebase for any style issues and ensure that all code follows the project's style guide.

---

By following this guide, you'll be well-prepared to contribute to `{{cookiecutter.project_slug}}`. Thank you for helping maintain and improve this project!
