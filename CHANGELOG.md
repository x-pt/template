# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2024-08-18

### Added

- [template] introduce cookiecutter, drop cargo-generate for project templates
- [python] introduce hatch and uv for python boilerplate

### Changed

- [cpp] update the CMakeLists.txt
- [python] update Dockerfile

## [0.1.1] - 2024-08-18

### Added

- [cpp] add `ccache` to speed up the compile-time
- [go] add GitHub Action jobs
- [all] add some default info to CHANGELOG template
- [template] introduce renovate
- [template] introduce release action
- [cpp] add renovate.json
- [template] introduce the issue template
- [template] introduce the pull request template

### Changed

- [python] replace {{project-name}} dir with src dir
- [cpp] image-building only triggers when a tag is pushed
- [cpp] remove audit action
- [cpp] keep Makefile even if it is a CMake project

### Fixed

## [0.1.0-beta] - 2022-09-21

### Added

- [cpp] distinguishes static and dynamic
- [cpp] distinguishes binary and library
- [all] add Dockerfile for each startup project
- [cpp] add example project [x-pt/example](https://github.com/x-pt/example)
- :sparkles: [template] support python project # THE VERY BEGINNING
- :sparkles: [template] support golang project # THE VERY BEGINNING
- :sparkles: [template] support cpp project    # ALMOST DONE
- :earth_asia: [template] `cargo generate gh:x-pt/template`
