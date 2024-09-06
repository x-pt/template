# ===================== Google Test Setup =====================

# Enable testing
enable_testing()

# Fetch and make available Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ===================== Test Sources =====================

# Collect test source files
file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# ===================== Test Executable =====================

# Create test executable
add_executable({{cookiecutter.project_slug}}-tests ${TEST_SOURCES})

# Set CUDA properties for the test executable
set_target_properties({{cookiecutter.project_slug}}-tests PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

# Include CUDA directories
target_include_directories({{cookiecutter.project_slug}}-tests
        PRIVATE
        ${CUDAToolkit_INCLUDE_DIRS}
)

# Link the test executable with Google Test, CUDA, and cuBLAS
target_link_libraries({{cookiecutter.project_slug}}-tests
        PRIVATE
        {{cookiecutter.package_name}}_lib
        GTest::gtest_main
        CUDA::cudart
        CUDA::cublas
)

# ===================== Test Discovery =====================

# Discover tests
include(GoogleTest)
gtest_discover_tests({{cookiecutter.project_slug}}-tests)
