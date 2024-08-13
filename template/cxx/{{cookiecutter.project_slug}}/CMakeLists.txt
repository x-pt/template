cmake_minimum_required(VERSION {{cookiecutter.cmake_min_version}})
project({{cookiecutter.project_slug}} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED {{cookiecutter.cxx_standard_required}})
set(CMAKE_CXX_EXTENSIONS {{cookiecutter.cxx_extensions_required}}) # enable compiler-specific features or not
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # clangd completion

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

{% if cookiecutter.project_type == "binary" -%}

    add_executable(${PROJECT_NAME} ${SOURCES})

{% else -%}

    {% if not cookiecutter.is_shared -%}
        # static library
        add_library({{cookiecutter.project_slug}} STATIC src/library.cpp)
    {% else -%}
        # shared library
        add_library({{cookiecutter.project_slug}} SHARED src/library.cpp)
    {% endif %}

{% endif %}

{% if cookiecutter.project_type == "binary" and not cookiecutter.is_shared -%}
    target_link_libraries(${PROJECT_NAME} PUBLIC "-static")
{% endif %}
