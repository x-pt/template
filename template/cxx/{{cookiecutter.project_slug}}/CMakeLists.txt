cmake_minimum_required(VERSION 3.28)
project({{cookiecutter.project_slug}} LANGUAGES CXX)

# C++ Standard settings
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED {{cookiecutter.cxx_standard_required}})
set(CMAKE_CXX_EXTENSIONS {{cookiecutter.cxx_extensions_required}})  # Enable or disable compiler-specific features

# Enable compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Source files
file(GLOB SOURCES CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.h ${CMAKE_SOURCE_DIR}/src/*.cpp)

# Speed up the compile by ccache
if(NOT MSVC)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
        set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
        set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
    endif()
endif()

# Specify architecture for macOS
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

{% if cookiecutter.cxx_project_type == "binary" %}
# Binary project setup
add_executable({{cookiecutter.project_slug}} ${SOURCES})

{% if cookiecutter.cxx_share_enabled == "STATIC" %}
# Additional settings for static binary on non-macOS platforms
if(NOT APPLE)
    target_link_libraries({{cookiecutter.project_slug}} PUBLIC "-static")
endif()
{% endif %}

{% else %}
# Library project setup
add_library({{cookiecutter.project_slug}} {{cookiecutter.cxx_share_enabled}} ${SOURCES})
{% endif %}
