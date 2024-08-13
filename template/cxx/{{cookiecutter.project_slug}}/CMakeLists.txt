cmake_minimum_required(VERSION {{cookiecutter.cmake_min_version}})
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

# Determine target type
set(IS_BINARY {{ cookiecutter.project_type == "binary" }})
set(IS_SHARED {{ cookiecutter.is_shared }})

if(IS_BINARY)
    add_executable(${PROJECT_NAME} ${SOURCES})
else()
    add_library({{cookiecutter.project_slug}} ${IS_SHARED} src/library.cpp)
endif()

# Link static libraries for binary projects if not shared
if(IS_BINARY AND NOT IS_SHARED)
    target_link_libraries(${PROJECT_NAME} PUBLIC "-static")
endif()
