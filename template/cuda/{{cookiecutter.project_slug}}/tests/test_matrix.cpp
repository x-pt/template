#include <gtest/gtest.h>
#include "matrix_add.h"
#include "matrix_mult.h"
#include <vector>

template <typename T>
class MatrixOperationsTest : public ::testing::Test {
protected:
    static constexpr int kMatrixSize = 2;
    static constexpr int kMatrixElements = kMatrixSize * kMatrixSize;

    std::vector<T> matrixA;
    std::vector<T> matrixB;
    std::vector<T> resultMatrix;

    void SetUp() override {
        // Initialize matrices A and B with test data
        matrixA = {1, 2, 3, 4};
        matrixB = {5, 6, 7, 8};
        resultMatrix.resize(kMatrixElements);
    }

    void verifyResult(const std::vector<T>& expected) {
        ASSERT_EQ(resultMatrix.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_NEAR(resultMatrix[i], expected[i], 1e-5)
                << "Mismatch at index " << i;
        }
    }
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MatrixOperationsTest, TestTypes);

TYPED_TEST(MatrixOperationsTest, AdditionTest) {
    addMatricesOnGPU(this->matrixA.data(), this->matrixB.data(), this->resultMatrix.data(),
                     this->kMatrixSize, this->kMatrixSize);

    std::vector<TypeParam> expectedSum = {6, 8, 10, 12};
    this->verifyResult(expectedSum);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationTest) {
    multiplyMatricesOnGPU(this->matrixA.data(), this->matrixB.data(), this->resultMatrix.data(),
                          this->kMatrixSize, this->kMatrixSize, this->kMatrixSize);

    std::vector<TypeParam> expectedProduct = {19, 22, 43, 50};
    this->verifyResult(expectedProduct);
}

TYPED_TEST(MatrixOperationsTest, NonSquareAdditionTest) {
    const int rows = 2;
    const int cols = 3;
    std::vector<TypeParam> nonSquareA = {1, 2, 3, 4, 5, 6};
    std::vector<TypeParam> nonSquareB = {7, 8, 9, 10, 11, 12};
    std::vector<TypeParam> nonSquareResult(rows * cols);

    addMatricesOnGPU(nonSquareA.data(), nonSquareB.data(), nonSquareResult.data(), rows, cols);

    std::vector<TypeParam> expectedSum = {8, 10, 12, 14, 16, 18};
    ASSERT_EQ(nonSquareResult.size(), expectedSum.size());
    for (size_t i = 0; i < expectedSum.size(); i++) {
        EXPECT_NEAR(nonSquareResult[i], expectedSum[i], 1e-5)
            << "Mismatch at index " << i;
    }
}

TYPED_TEST(MatrixOperationsTest, NonSquareMultiplicationTest) {
    const int rowsA = 2;
    const int colsA = 3;
    const int colsB = 2;
    std::vector<TypeParam> nonSquareA = {1, 2, 3, 4, 5, 6};
    std::vector<TypeParam> nonSquareB = {7, 8, 9, 10, 11, 12};
    std::vector<TypeParam> nonSquareResult(rowsA * colsB);

    multiplyMatricesOnGPU(nonSquareA.data(), nonSquareB.data(), nonSquareResult.data(),
                          rowsA, colsA, colsB);

    std::vector<TypeParam> expectedProduct = {58, 64, 139, 154};
    ASSERT_EQ(nonSquareResult.size(), expectedProduct.size());
    for (size_t i = 0; i < expectedProduct.size(); i++) {
        EXPECT_NEAR(nonSquareResult[i], expectedProduct[i], 1e-5)
            << "Mismatch at index " << i;
    }
}
