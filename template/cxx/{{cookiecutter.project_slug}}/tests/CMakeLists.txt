# Enable testing functionality.
enable_testing()

# ===== GoogleTest Dependency =====
# Ensure googletest is only fetched and built once.
if(NOT TARGET GTest::gtest)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    )
    # Force GoogleTest to use shared runtime libraries (useful on Windows).
    set(gtest_force_shared_crt ON CACHE BOOL "Use shared CRT" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ===== Test Sources =====
file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Create the test executable and link dependencies.
add_executable(${PROJECT_NAME}-tests ${TEST_SOURCES})
target_link_libraries(${PROJECT_NAME}-tests PRIVATE
    GTest::gtest_main
    ${PROJECT_NAME}_lib
)

# Use strict warnings for tests if needed.
if(MSVC)
    target_compile_options(${PROJECT_NAME}-tests PRIVATE /W4 /permissive-)
else()
    target_compile_options(${PROJECT_NAME}-tests PRIVATE -Wall -Wextra)
endif()

# ===== Test Discovery =====
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}-tests)
