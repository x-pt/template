# Changelog

All notable changes to this project will be documented in this file.

## [unreleased]

### 🚀 Features

- *(cxx)* Optimize the CMakeLists
- *(cxx)* Optimize the CMakeLists
- *(cxx)* Introduce xmake option
- *(cxx)* Setup the output dir for binary and lib

### 🚜 Refactor

- *(cxx)* Rearrange the CMakeLists

### 📚 Documentation

- Update CHANGELOG
- Update some doc
- Add code_of_conduct
- *(python)* Update the README.md
- *(python)* Split the development guide
- *(python)* Require python version by the input
- *(python)* Update the development.md
- *(cxx)* Remove the rust info
- *(cxx)* Update the README.md

### 🎨 Styling

- Using dash instead of asterisk

### ⚙️ Miscellaneous Tasks

- Rearrange the template inputs location
- Update compose
- Add labeler for issues
- Update issue labeler match rules
- Fix enable-versioned-regex not found
- Make some minor changes
- Add year to license dynamically
- Use local time for license date
- Update the pre-commit version

### Build

- *(go)* Update the package version
- Remove version from the docker compose
- *(python)* Use uv for all scenario
- *(python)* Only install uv at Dockerfile
- *(python)* Add UV_INDEX_URL at Dockerfile
- *(python)* Use debian bookworm
- Introduce git-cliff to generate the changelog

## [1.0.0] - 2024-08-17

### 🚀 Features

- Introduce hatch and uv for python
- Update python template
- Introduce rhai for preprocessing
- Update python boilerplate
- Update python Dockerfile
- Use the specified python version
- *(py)* Update Dockerfile and compose
- *(py)* Update README and refactor Dockerfile and compose

### 🐛 Bug Fixes

- Msvc cannot recognize the header file with '-'

### 🚜 Refactor

- Use cookiecutter instead of cargo-generate
- *(ongoing)* Introduce cookiecutter
- *(ongoing)* Change the location of py cookiecutter
- *(py)* Introduce cookiecutter
- *(init)* Introduce cookiecutter for cxx and golang
- Introduce cookiecutter for cxx and golang
- Minor changes
- Optimize the cmakelist

### 🎨 Styling

- Align the format

### 🧪 Testing

- Add some test cases

### ⚙️ Miscellaneous Tasks

- Update workflow
- Some minor changes
- Update README
- Update README
- Make some minor changes
- Make some minor changes
- Make some minor changes
- Make some minor changes
- Update workflow
- Fix workflow failure

## [0.1.1] - 2024-04-16

### 🚀 Features

- *(github)* Add contributing and issuing template
- *(cmake)* Update the CMakeLists.txt and add Makefile
- *(Makefile)* Add clean command
- *(docker)* Add github action to build docker image
- *(docker)* Add the prefix for github docker image
- *(cmake)* DO NOT IGNORE Makefile
- *(action)* Remove audit
- *(action)* Change the logic of docker build
- *(action)* Add renovate for cpp
- *(action)* Add release and renovate action
- *(cargo-gen)* Add much more support for python
- *(docker)* Build image only when tagged like as v0.1.0
- *(template)* Change {{project-name}} dir to src in python project
- *(changelog)* Add some default info in CHANGELOG.md
- *(go)* Add github actions
- Add .editorconfig
- Add ccache to speed up the compile
- *(cxx)* Replace the cmake build way
- *(cxx)* Add some files to .dockerignore
- *(py)* Add .dockerignore
- Update the docker buildx version
- Update the docker buildx version
- Update the docker buildx version
- Remove macos verify from cxx
- *(go)* Add some options
- Update cxx, go, py template
- Update code of conduct
- Update the docker github action
- *(py)* Add src-layout
- Update go proj template
- Go template with cobra and viper
- *(go)* Some minor changes
- *(go)* Update Dockerfile and github action
- Update the github action cd.yml in go and cxx template
- Update the github action cd.yml in py template
- Introduce pre-commit for all templates
- Update python template ci

### 🐛 Bug Fixes

- Failed to rye init
- Failed to make build for example go on github ci

### ⚙️ Miscellaneous Tasks

- Add missing "bin_type"
- *(support)* Add jetbrains badge
- Some minor changes
- Update checkout to v4
- Change the typo
- Add Makefile tab rule in editorconfig
- Some minor changes
- Use make build and test
- Downgrade the cargo-generate-action
- Some minor changes
- Rename pre-commit file
- Gen go and py example
- Continue to fix project name
- Update liquid syntax

### Build

- *(deps)* Bump actions/checkout from 2 to 3
- *(deps)* Bump cargo-generate/cargo-generate-action
- *(deps)* Bump cargo-generate/cargo-generate-action
- *(deps)* Bump actions/checkout from 3 to 4
- *(deps)* Bump cargo-generate/cargo-generate-action
- *(deps)* Bump peaceiris/actions-gh-pages from 3 to 4
- *(deps)* Bump softprops/action-gh-release from 1 to 2
- *(deps)* Bump cargo-generate/cargo-generate-action

## [0.1.0-beta] - 2022-09-20

### 🚀 Features

- *(init)* Cargo generate cpp project
- *(classify)* Bin and lib(shared or static)
- Add .editorconfig for new project
- :art: optimize the control flow
- *(template)* Add golang and python support
- *(template)* Update cxx CMakeLists.txt
- *(template)* Add Dockerfile
- *(docker)* Support Dockerfile
- *(docker)* Optimize the Dockerfile
- *(docker)* Add static binary compile
- *(docker)* Enable crb repo
- *(docker)* Distinguish static and dynamic binary on Dockerfile
- Add changelog

### 🚜 Refactor

- *(rename)* Org name cxx-gh to gh-proj
- *(rename)* Gh-proj to x-pt
- *(dir)* Rearrange the hierarchy

### 📚 Documentation

- Fix some legacy info

### ⚙️ Miscellaneous Tasks

- *(init)* Update deployment
- *(init)* Remove the checkout of example repo
- *(init)* Include .github dir and exclude .nojekyll file
- *(init)* Remove .nojekyll file
- Replace the commit message
- Enable jekyll to remove .nojekyll
- Rename some variables
- Fix ignore not work
- Add missing fields for cd
- Update ci.yml for cpp project
- Update cd.yml for cpp project
- Update cd.yml for cpp project
- Update cd.yml for cpp project
- Fix ci error
- Remove build action

<!-- generated by git-cliff -->
