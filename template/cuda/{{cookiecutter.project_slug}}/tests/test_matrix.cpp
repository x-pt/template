#include <gtest/gtest.h>
#include "matrix_add.h"
#include "matrix_mult.h"

class MatrixOperationsTest : public ::testing::Test {
protected:
    static constexpr int kMatrixSize = 2;
    static constexpr int kMatrixElements = kMatrixSize * kMatrixSize;

    float matrixA[kMatrixElements];
    float matrixB[kMatrixElements];
    float resultMatrix[kMatrixElements];

    void SetUp() override {
        // Initialize matrices A and B with test data
        float dataA[kMatrixElements] = {1, 2, 3, 4};
        float dataB[kMatrixElements] = {5, 6, 7, 8};
        std::copy(std::begin(dataA), std::end(dataA), std::begin(matrixA));
        std::copy(std::begin(dataB), std::end(dataB), std::begin(matrixB));
    }

    void verifyResult(const float expected[kMatrixElements]) {
        for (int i = 0; i < kMatrixElements; i++) {
            EXPECT_FLOAT_EQ(resultMatrix[i], expected[i])
                << "Mismatch at index " << i;
        }
    }
};

TEST_F(MatrixOperationsTest, AdditionTest) {
    addMatricesOnGPU(matrixA, matrixB, resultMatrix, kMatrixSize, kMatrixSize);

    float expectedSum[kMatrixElements] = {6, 8, 10, 12};
    verifyResult(expectedSum);
}

TEST_F(MatrixOperationsTest, MultiplicationTest) {
    multiplyMatricesOnGPU(matrixA, matrixB, resultMatrix, kMatrixSize, kMatrixSize, kMatrixSize);

    float expectedProduct[kMatrixElements] = {19, 22, 43, 50};
    verifyResult(expectedProduct);
}
