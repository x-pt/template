cmake_minimum_required(VERSION {{cmake-min-version}})
project({{project-name}})

set(CMAKE_CXX_STANDARD {{cxx-standard}})

add_executable({{project-name}} main.cpp)
