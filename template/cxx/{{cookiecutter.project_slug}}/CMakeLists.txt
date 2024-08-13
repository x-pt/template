cmake_minimum_required(VERSION {{cmake_min_version}})
project({{project-name}} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD {{cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED {{cxx_standard_required}})
set(CMAKE_CXX_EXTENSIONS {{cxx_extensions}}) # enable compiler-specific features or not
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

{% if cxx_proj_type == "bin" -%}

add_executable(${PROJECT_NAME} ${SOURCES})

{% if cxx_bin_type == "static" -%}

target_link_libraries(${PROJECT_NAME} PUBLIC "-static")

{%- endif %}

{%- elsif cxx_lib_type == "static" -%}

# static
add_library({{project-name}} src/library.cpp)

{%- else -%}

# shared
add_library({{project-name}} SHARED src/library.cpp)

{%- endif %}
