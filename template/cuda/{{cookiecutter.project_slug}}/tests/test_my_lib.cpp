#include "gtest/gtest.h"
#include <vector> // For std::vector
#include "matrix_add.h" // For addMatricesOnGPU
#include "my_lib.h" // For existing arithmetic tests

// Group: Add Function Tests
TEST(AddTest, HandlesZeroInputs) {
    EXPECT_EQ(add(0, 0), 0);
}

TEST(AddTest, HandlesPositiveInputs) {
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(10, 5), 15);
}

TEST(AddTest, HandlesNegativeInputs) {
    EXPECT_EQ(add(-1, -2), -3);
    EXPECT_EQ(add(-5, 5), 0);
}

// Group: Subtract Function Tests
TEST(SubtractTest, HandlesZeroInputs) {
    EXPECT_EQ(sub(0, 0), 0);
}

TEST(SubtractTest, HandlesPositiveInputs) {
    EXPECT_EQ(sub(10, 5), 5);
    EXPECT_EQ(sub(5, 10), -5);
}

TEST(SubtractTest, HandlesNegativeInputs) {
    EXPECT_EQ(sub(-10, -5), -5);
    EXPECT_EQ(sub(-5, -10), 5);
}

// Group: Multiply Function Tests
TEST(MultiplyTest, HandlesZeroInputs) {
    EXPECT_EQ(mul(0, 0), 0);
}

TEST(MultiplyTest, HandlesPositiveInputs) {
    EXPECT_EQ(mul(2, 3), 6);
    EXPECT_EQ(mul(6, 3), 18);
}

TEST(MultiplyTest, HandlesNegativeInputs) {
    EXPECT_EQ(mul(-2, -3), 6);
    EXPECT_EQ(mul(-2, 3), -6);
}

// Group: Divide Function Tests
TEST(DivideTest, HandlesZeroDivision) {
    EXPECT_THROW(divide(10.0, 0.0), std::invalid_argument);
}

TEST(DivideTest, HandlesPositiveInputs) {
    EXPECT_NEAR(divide(10.0, 3.0), 3.33333, 1e-5);
    EXPECT_NEAR(divide(10.0, 2.0), 5.0, 1e-5);
}

TEST(DivideTest, HandlesNegativeInputs) {
    EXPECT_NEAR(divide(-10.0, 2.0), -5.0, 1e-5);
}

// Group: CUDA Function Tests
TEST(MatrixAddGPUTest, HandlesBasicAddition) {
    const int rows = 2;
    const int cols = 2;
    const int N = rows * cols;

    std::vector<float> h_A = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> h_B = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> h_C_result(N);
    std::vector<float> h_C_expected = {6.0f, 8.0f, 10.0f, 12.0f};

    // Call the CUDA function to add matrices on the GPU
    addMatricesOnGPU(h_A.data(), h_B.data(), h_C_result.data(), rows, cols);

    // Verify the result
    ASSERT_EQ(h_C_result.size(), h_C_expected.size());
    for (size_t i = 0; i < h_C_expected.size(); ++i) {
        EXPECT_NEAR(h_C_result[i], h_C_expected[i], 1e-5f);
    }
}
