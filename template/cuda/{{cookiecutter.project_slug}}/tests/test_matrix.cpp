#include <gtest/gtest.h>

#include <vector>

#include "matrix_add.h"
#include "matrix_mult.h"

// Template-based parameterized test for matrix operations
template <typename T>
class MatrixOperationsTest : public testing::Test {
protected:
    // Static constants for default matrix configuration
    static constexpr int kDefaultMatrixSize = 2;
    static constexpr int kDefaultMatrixElements = kDefaultMatrixSize * kDefaultMatrixSize;

    // Helper function to create a matrix from initializer list
    std::vector<T> createMatrix(std::initializer_list<T> values) { return std::vector<T>(values); }

    // Helper function to verify matrix calculation results
    static void verifyResult(const std::vector<T>& result, const std::vector<T>& expected) {
        // Check matrix size
        ASSERT_EQ(result.size(), expected.size()) << "Result matrix size does not match expected matrix size";

        // Compare elements with near-equality
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_NEAR(result[i], expected[i], 1e-5) << "Mismatch at index " << i << ": expected " << expected[i] << ", got " << result[i];
        }
    }

    // SetUp method to print CUDA versions before each test
    void SetUp() override {
        int runtime_version, driver_version;

        cudaRuntimeGetVersion(&runtime_version);
        std::cout << "CUDA Runtime Version: "
                  << runtime_version / 1000 << "." << (runtime_version % 1000) / 10 << std::endl;

        cudaDriverGetVersion(&driver_version);
        std::cout << "CUDA Driver Version: "
                  << driver_version / 1000 << "." << (driver_version % 1000) / 10 << std::endl;
    }
};

// Register type-parameterized test suite
TYPED_TEST_SUITE_P(MatrixOperationsTest);

// Test case 1: Square matrix addition
TYPED_TEST_P(MatrixOperationsTest, SquareMatrixAddition) {
    constexpr int size = this->kDefaultMatrixSize;

    // Prepare test data
    auto matrixA = this->createMatrix({1, 2, 3, 4});
    auto matrixB = this->createMatrix({5, 6, 7, 8});
    std::vector<TypeParam> resultMatrix(size * size);

    // Call GPU matrix addition function
    addMatricesOnGPU(matrixA.data(), matrixB.data(), resultMatrix.data(), size, size);

    // Verify result
    auto expectedSum = this->createMatrix({6, 8, 10, 12});
    this->verifyResult(resultMatrix, expectedSum);
}

// Test case 2: Square matrix multiplication
TYPED_TEST_P(MatrixOperationsTest, SquareMatrixMultiplication) {
    constexpr int size = this->kDefaultMatrixSize;

    // Prepare test data
    auto matrixA = this->createMatrix({1, 2, 3, 4});
    auto matrixB = this->createMatrix({5, 6, 7, 8});
    std::vector<TypeParam> resultMatrix(size * size);

    // Call GPU matrix multiplication function
    multiplyMatricesOnGPU(matrixA.data(), matrixB.data(), resultMatrix.data(), size, size, size);

    // Verify result
    auto expectedProduct = this->createMatrix({19, 22, 43, 50});
    this->verifyResult(resultMatrix, expectedProduct);
}

// Test case 3: Non-square matrix addition
TYPED_TEST_P(MatrixOperationsTest, NonSquareMatrixAddition) {
    constexpr int rows = 2;
    constexpr int cols = 3;

    // Prepare test data
    auto nonSquareA = this->createMatrix({1, 2, 3, 4, 5, 6});
    auto nonSquareB = this->createMatrix({7, 8, 9, 10, 11, 12});
    std::vector<TypeParam> nonSquareResult(rows * cols);

    // Call GPU matrix addition function
    addMatricesOnGPU(nonSquareA.data(), nonSquareB.data(), nonSquareResult.data(), rows, cols);

    // Verify result
    auto expectedSum = this->createMatrix({8, 10, 12, 14, 16, 18});
    this->verifyResult(nonSquareResult, expectedSum);
}

// Test case 4: Non-square matrix multiplication
TYPED_TEST_P(MatrixOperationsTest, NonSquareMatrixMultiplication) {
    constexpr int rowsA = 2;
    constexpr int colsA = 3;
    constexpr int colsB = 2;

    // Prepare test data
    auto nonSquareA = this->createMatrix({1, 2, 3, 4, 5, 6});
    auto nonSquareB = this->createMatrix({7, 8, 9, 10, 11, 12});
    std::vector<TypeParam> nonSquareResult(rowsA * colsB);

    // Call GPU matrix multiplication function
    multiplyMatricesOnGPU(nonSquareA.data(), nonSquareB.data(), nonSquareResult.data(), rowsA, colsA, colsB);

    // Verify result
    auto expectedProduct = this->createMatrix({58, 64, 139, 154});
    this->verifyResult(nonSquareResult, expectedProduct);
}

// Register test cases
REGISTER_TYPED_TEST_SUITE_P(MatrixOperationsTest, SquareMatrixAddition, SquareMatrixMultiplication, NonSquareMatrixAddition, NonSquareMatrixMultiplication);

// Specify test types
using TestTypes = testing::Types<float, double>;
INSTANTIATE_TYPED_TEST_SUITE_P(MatrixOps, MatrixOperationsTest, TestTypes);
