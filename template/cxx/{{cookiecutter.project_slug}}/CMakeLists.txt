cmake_minimum_required(VERSION {{cookiecutter.cxx_cmake_version}})
project({{cookiecutter.project_slug}} LANGUAGES CXX)

# C++ Standard settings
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED {{cookiecutter.cxx_standard_required}})
set(CMAKE_CXX_EXTENSIONS {{cookiecutter.cxx_extensions_required}})  # enable compiler-specific features or not

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

{% if cookiecutter.cxx_project_type == "binary" %}
add_executable({{cookiecutter.project_slug}} ${SOURCES})
{% else %}
add_library({{cookiecutter.project_slug}} {{cookiecutter.cxx_share_enabled}} src/library.cpp)
{% endif %}

{% if cookiecutter.cxx_project_type == "binary" and not cookiecutter.cxx_share_enabled %}
target_link_libraries({{cookiecutter.project_slug}} PUBLIC "-static")
{% endif %}
