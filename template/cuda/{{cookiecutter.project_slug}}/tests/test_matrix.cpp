#include <gtest/gtest.h>
#include "matrix_add.h"
#include "matrix_mult.h"

TEST(MatrixOperations, AddTest) {
    const int rows = 2;
    const int cols = 2;
    float A[rows * cols] = {1, 2, 3, 4};
    float B[rows * cols] = {5, 6, 7, 8};
    float C[rows * cols] = {0};

    matrixAddHost(A, B, C, rows, cols);

    float expected[rows * cols] = {6, 8, 10, 12};
    for (int i = 0; i < rows * cols; i++) {
        EXPECT_FLOAT_EQ(C[i], expected[i]);
    }
}

TEST(MatrixOperations, MultTest) {
    const int rowsA = 2;
    const int colsA = 2;
    const int colsB = 2;

    float A[rowsA * colsA] = {1, 2, 3, 4};
    float B[colsA * colsB] = {5, 6, 7, 8};
    float C[rowsA * colsB] = {0};

    matrixMultHost(A, B, C, rowsA, colsA, colsB);

    float expected[rowsA * colsB] = {19, 22, 43, 50};
    for (int i = 0; i < rowsA * colsB; i++) {
        EXPECT_FLOAT_EQ(C[i], expected[i]);
    }
}
