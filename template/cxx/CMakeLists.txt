cmake_minimum_required(VERSION {{cmake_min_version}})
project({{project-name}} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD {{cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED {{cxx_standard_required}})
set(CMAKE_CXX_EXTENSIONS {{cxx_extensions}}) # enable compiler-specific features or not
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # clangd completion

file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")

{% if proj_type == "bin" -%}
add_executable(${PROJECT_NAME} ${SOURCES})
{%- elsif lib_type == "static" -%}
# static
add_library({{project-name}} src/library.cpp)
{%- else -%}
# shared
add_library({{project-name}} SHARED src/library.cpp)
{%- endif %}
