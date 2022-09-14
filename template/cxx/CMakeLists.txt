cmake_minimum_required(VERSION {{cmake_min_version}})
project({{project-name}})

set(CMAKE_CXX_STANDARD {{cxx_standard_version}})

{% if proj_type == "bin" -%}
add_executable({{project-name}} src/main.cpp)
{%- elsif lib_type == "static" -%}
# static
add_library({{project-name}} src/library.cpp)
{%- else -%}
# shared
add_library({{project-name}} SHARED src/library.cpp)
{%- endif %}
