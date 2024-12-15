cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

# Project setup with version, description, homepage, and languages.
project(
    {{cookiecutter.project_slug}}
    VERSION {{cookiecutter.project_version}}
    DESCRIPTION "{{cookiecutter.project_desc}}"
    HOMEPAGE_URL "https://github.com/{{cookiecutter.__gh_slug}}"
    LANGUAGES CXX
)

# ===== Project-wide Configuration =====
# Extend CMake module path for custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# C++ Standard and Compilation Settings
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type with configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build" FORCE)
endif()

# Standardized output directories with generator expressions
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

# ===== Dependency Management =====
include(FetchContent)

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY "https://github.com/gabime/spdlog.git"
    GIT_TAG "v1.15.0"
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(spdlog)

# ===== Source File Discovery =====
# Centralized source file discovery with better organization
set(PROJECT_SOURCE_DIRS
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party
)

# Advanced file discovery with more robust configuration
foreach(DIR ${PROJECT_SOURCE_DIRS})
    file(GLOB_RECURSE CURRENT_HEADERS
        CONFIGURE_DEPENDS
        ${DIR}/*.h
        ${DIR}/*.hpp
        ${DIR}/*.hxx
    )
    file(GLOB_RECURSE CURRENT_SOURCES
        CONFIGURE_DEPENDS
        ${DIR}/*.cpp
        ${DIR}/*.cxx
        ${DIR}/*.cc
    )

    list(APPEND PROJECT_HEADERS ${CURRENT_HEADERS})
    list(APPEND PROJECT_SOURCES ${CURRENT_SOURCES})
endforeach()

# Remove duplicates to prevent potential compilation issues
list(REMOVE_DUPLICATES PROJECT_HEADERS)
list(REMOVE_DUPLICATES PROJECT_SOURCES)

# ===== Library Target =====
# Create modular library with comprehensive configurations
add_library(${PROJECT_NAME}_lib STATIC
    ${PROJECT_HEADERS}
    ${PROJECT_SOURCES}
)

# Intelligent library linking and include management
target_link_libraries(${PROJECT_NAME}_lib
    PUBLIC
    spdlog::spdlog
)

# Advanced include directory management
target_include_directories(${PROJECT_NAME}_lib
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/third_party>
)

# ===== Compiler-Specific Optimizations =====
# Robust compiler detection and configuration
if(MSVC)
    target_compile_options(${PROJECT_NAME}_lib PRIVATE
        /W4           # Higher warning level
        /permissive-  # Strict standard conformance
        /utf-8        # UTF-8 source and execution character sets
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${PROJECT_NAME}_lib PRIVATE
        -Wall         # Enable all warnings
        -Wextra       # Additional warnings
        -Werror       # Treat warnings as errors
        -Wpedantic    # Enforce strict ISO C++ standards
    )
endif()

# ===== Executable Target =====
add_executable(${PROJECT_NAME}
    ${CMAKE_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${PROJECT_NAME}_lib
)

# ===== Testing =====
add_subdirectory(tests)
