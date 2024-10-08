# Specify the minimum version of CMake required
cmake_minimum_required(VERSION 3.25)

# Define the project
project({{cookiecutter.project_slug}}
        VERSION {{cookiecutter.project_version}}
        DESCRIPTION "{{cookiecutter.project_desc}}"
        HOMEPAGE_URL "https://github.com/{{cookiecutter.__gh_slug}}"
        LANGUAGES CXX CUDA
)

# ===================== Build Settings =====================

# Set CXX standards
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standards
set(CMAKE_CUDA_STANDARD {{cookiecutter.cuda_standard_version}})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Set compile options
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # Enable compile commands for clangd

# Set architecture for macOS (if applicable)
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# ===================== Output Directories =====================

# Use generator expressions for output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

# ===================== Source Files =====================

# Collect source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/*.cpp
        ${CMAKE_SOURCE_DIR}/src/*.cu
)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/include/*.h
)

# ===================== Library Target =====================

# Create library target
add_library({{cookiecutter.package_name}}_lib ${SOURCES} ${HEADERS})

# Set include directories for the library
target_include_directories({{cookiecutter.package_name}}_lib
        PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Link CUDA libraries to the library target
target_link_libraries({{cookiecutter.package_name}}_lib
        PUBLIC
        CUDA::cudart
        CUDA::cublas
)

# Set CUDA properties for the library
set_target_properties({{cookiecutter.package_name}}_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# ===================== Executable Target =====================

# Create executable target
add_executable({{cookiecutter.project_slug}} ${CMAKE_SOURCE_DIR}/src/main.cpp)

# Link library and CUDA libraries to the executable
target_link_libraries({{cookiecutter.project_slug}}
        PRIVATE
        {{cookiecutter.package_name}}_lib
        CUDA::cudart
        CUDA::cublas
)

# ===================== Tests =====================

# Add tests subdirectory
add_subdirectory(tests)
