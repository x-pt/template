#include "gtest/gtest.h"
#include "my_lib.h"

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
