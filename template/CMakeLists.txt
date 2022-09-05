cmake_minimum_required(VERSION {{cmake-min-version}})
project({{project-name}})

set(CMAKE_CXX_STANDARD {{cxx-standard}})

{% if project-type == "bin" %}
add_executable({{project-name}} src/main.cpp)
{% else if lib-type == "static" %}
# static
add_library({{project-name}} src/library.cpp)
{% else %}
# shared
add_library({{project-name}} SHARED src/library.cpp)
{% endif %}
