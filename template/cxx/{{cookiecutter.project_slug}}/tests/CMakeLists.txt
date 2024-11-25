# Enable testing
enable_testing()

# Fetch and make available Google Test
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Collect test source files
file(GLOB_RECURSE TEST_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create test executable
add_executable(${PROJECT_NAME}-tests ${TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}-tests PRIVATE
        GTest::gtest_main
        ${PROJECT_NAME}_lib
)

# Enable test discovery
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}-tests)
