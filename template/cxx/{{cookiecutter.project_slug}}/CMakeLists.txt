cmake_minimum_required(VERSION 3.25)
project({{cookiecutter.project_slug}}
        VERSION {{cookiecutter.project_version}}
        DESCRIPTION "{{cookiecutter.project_desc}}"
        HOMEPAGE_URL "https://github.com/{{cookiecutter.__gh_slug}}"
        LANGUAGES CXX
)

# ===== Custom Modules =====
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# ===== Build Settings =====
set(CMAKE_CXX_STANDARD {{cookiecutter.cxx_standard_version}})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/bin>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "$<1:${CMAKE_BINARY_DIR}/lib>")

# ===== Dependencies =====
include(FetchContent)

# spdlog
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.15.0
)
FetchContent_MakeAvailable(spdlog)

# third_party
add_subdirectory(third_party)

# Required libraries
# find_package(MPI REQUIRED)
# find_package(XXX REQUIRED)

# ===== Source Configuration =====
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
        ${SRC_DIR}/*.cpp
)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS
        ${INCLUDE_DIR}/*.h
)

# ===== Library Targets =====
add_library(${PROJECT_NAME}_lib ${SOURCES} ${HEADERS})
target_include_directories(${PROJECT_NAME}_lib
        PUBLIC
        $<BUILD_INTERFACE:${INCLUDE_DIR}>
        spdlog::spdlog
)

# ===== Executable Targets =====
add_executable(${PROJECT_NAME} ${SRC_DIR}/main.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_NAME}_lib)

# ===== Tests =====
add_subdirectory(tests)
