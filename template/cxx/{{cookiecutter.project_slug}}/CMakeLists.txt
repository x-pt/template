cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Project setup with version, description, homepage, and languages.
project(
    {{cookiecutter.project_slug}}
    VERSION {{cookiecutter.project_version}}
    DESCRIPTION "{{cookiecutter.project_desc}}"
    HOMEPAGE_URL "https://github.com/{{cookiecutter.__gh_slug}}"
    LANGUAGES CXX
)

# ===== Custom Modules =====
# Extend CMake module path for additional custom modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ===== Build Settings =====
# Specify C++ standard and other build configurations.
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set default build type to Release if not specified.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Configure output directories for generated binaries and libraries.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

# ===== Dependencies =====
include(FetchContent)

# Dependency: spdlog (logging library)
find_package(spdlog 1.15.0 QUIET)
if(NOT spdlog_FOUND)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
        GIT_TAG "v1.15.0"
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# Add third-party libraries directory if applicable.
add_subdirectory(third_party)

# Example of adding more required libraries (commented out).
# find_package(MPI REQUIRED)
# find_package(XXX REQUIRED)

# ===== Source Configuration =====
# Define source and include directories for the project.
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Recursively find all source and header files in respective directories.
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    ${SRC_DIR}/*.cpp
)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
    ${INCLUDE_DIR}/*.h
)

# ===== Library Targets =====
# Create a library target for project sources and headers.
add_library(${PROJECT_NAME}_lib ${SOURCES} ${HEADERS})

# Link library with required dependencies.
target_link_libraries(${PROJECT_NAME}_lib
    PUBLIC
    spdlog::spdlog   # Logging library
    httplib          # HTTP/HTTPS library (example, ensure this is available)
    cxxopts          # Command-line parsing library (example)
)

# Set include directories for the library.
target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
    $<BUILD_INTERFACE:${INCLUDE_DIR}>
)

# ===== Compiler Options =====
# Compiler-specific options for warnings and standards compliance.
if(MSVC)
    target_compile_options(${PROJECT_NAME}_lib PRIVATE
        /W4           # Higher warning level
        /permissive-  # Enforce strict standard conformance
    )
else()
    target_compile_options(${PROJECT_NAME}_lib PRIVATE
        -Wall         # Enable all warnings
        -Wextra       # Additional warnings
        -Werror       # Treat warnings as errors
    )
endif()

# ===== Executable Targets =====
# Define the main executable and link it with the library.
add_executable(${PROJECT_NAME} ${SRC_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

# ===== Tests =====
# Add subdirectory for test-related configurations.
add_subdirectory(tests)
