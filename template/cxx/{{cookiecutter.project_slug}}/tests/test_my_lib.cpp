#include "gtest/gtest.h"
#include "my_lib.h"

TEST(AddTest, PositiveNumbers)
{
    EXPECT_EQ(add(1, 2), 3);
    EXPECT_EQ(add(10, 5), 15);
}

TEST(AddTest, NegativeNumbers)
{
    EXPECT_EQ(add(-1, -2), -3);
    EXPECT_EQ(add(-5, 5), 0);
}

TEST(SubTest, PositiveNumbers)
{
    EXPECT_EQ(sub(10, 5), 5);
    EXPECT_EQ(sub(5, 10), -5);
}

TEST(SubTest, NegativeNumbers)
{
    EXPECT_EQ(sub(-10, -5), -5);
    EXPECT_EQ(sub(-5, -10), 5);
}
