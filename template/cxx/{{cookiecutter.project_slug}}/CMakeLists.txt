cmake_minimum_required(VERSION 3.28)
project({{cookiecutter.project_slug}}
    VERSION 0.0.1
    DESCRIPTION "{{cookiecutter.project_desc}}"
    HOMEPAGE_URL "https://github.com/{{cookiecutter.__gh_slug}}"
    LANGUAGES CXX
)

# C++ Standard settings
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Enable or disable compiler-specific features

# Enable compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Specify architecture for macOS
if(APPLE)
    set(CMAKE_OSX_ARCHITECTURES "arm64")
endif()

# Set output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

# Set output directory for libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Set output directory for archives
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# Collect source files
file(GLOB_RECURSE SOURCES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE HEADERS ${CMAKE_SOURCE_DIR}/include/*.h)

# Create library target
add_library({{cookiecutter.package_name}}_lib ${SOURCES} ${HEADERS})
target_include_directories({{cookiecutter.package_name}}_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Create executable target
add_executable({{cookiecutter.project_slug}} ${CMAKE_SOURCE_DIR}/src/main.cpp)
target_link_libraries({{cookiecutter.project_slug}} PRIVATE {{cookiecutter.package_name}}_lib)

# Add tests subdirectory
add_subdirectory(tests)
