cmake_minimum_required(VERSION 3.25)

project({{cookiecutter.project_slug}}
        VERSION {{cookiecutter.project_version}}
        DESCRIPTION "{{cookiecutter.project_desc}}"
        HOMEPAGE_URL "https://github.com/{{cookiecutter.__gh_slug}}"
        LANGUAGES CXX CUDA
)

# Custom modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Build settings
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD {{cookiecutter.cuda_standard_version}})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type configuration
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find dependencies
find_package(CUDAToolkit REQUIRED)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

# Source files configuration
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        ${SRC_DIR}/*.cpp
        ${SRC_DIR}/*.cu
)
list(FILTER SOURCES EXCLUDE REGEX ".*main\\.cpp$")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
        ${INCLUDE_DIR}/*.h
)

# Library target
add_library(${PROJECT_NAME}_lib ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME}_lib
        PUBLIC
        $<BUILD_INTERFACE:${INCLUDE_DIR}>
)
target_link_libraries(${PROJECT_NAME}_lib
        PUBLIC
        CUDA::cudart
        CUDA::cublas
)
set_target_properties(${PROJECT_NAME}_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
)

# Executable target
add_executable(${PROJECT_NAME} ${SRC_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

# Tests
add_subdirectory(tests)
