// src/main.cpp
// This application demonstrates a simple CUDA matrix addition.
// It initializes two matrices on the host (CPU), transfers them to the GPU,
// performs element-wise addition on the GPU, and copies the result back to the host.

#include <iostream>  // For std::cout, std::endl
#include <vector>    // For std::vector
#include <iomanip>   // For std::fixed and std::setprecision to format output

// Contains the declaration for the addMatricesOnGPU CUDA function
#include "matrix_add.h"

// Optional: For my_lib.h arithmetic (can be removed if not used for other demos)
// #include "my_lib.h"

// Helper function to print a matrix to the console.
// Formats the output for better readability.
template <typename T>
void printMatrix(const std::vector<T>& matrix, int rows, int cols, const std::string& title) {
    std::cout << title << " (" << rows << "x" << cols << "):" << std::endl;
    if (matrix.empty()) {
        std::cout << "  [Empty Matrix]" << std::endl;
        return;
    }
    // Loop through each row
    for (int i = 0; i < rows; ++i) {
        std::cout << "  [";
        // Loop through each column in the current row
        for (int j = 0; j < cols; ++j) {
            // Output matrix element with fixed precision
            std::cout << std::fixed << std::setprecision(1) << matrix[i * cols + j]
                      // Add a comma and space unless it's the last element in the row
                      << (j == cols - 1 ? "" : ", ");
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl; // Extra newline for spacing
}

int main() {
    std::cout << "CUDA Matrix Addition Demonstration" << std::endl;
    std::cout << "---------------------------------" << std::endl << std::endl;

    // Define matrix dimensions: rows x cols
    const int rows = 3;
    const int cols = 4;
    const int numElements = rows * cols; // Total number of elements in each matrix

    // Initialize host matrices (CPU memory) using std::vector.
    // std::vector manages its own memory, ensuring it's properly allocated and deallocated.
    std::vector<float> hostMatrixA(numElements);
    std::vector<float> hostMatrixB(numElements);
    std::vector<float> hostResultMatrixC(numElements); // To store the result C = A + B

    // Populate matrices A and B with some sample values.
    std::cout << "Initializing host matrices A and B..." << std::endl;
    for (int i = 0; i < numElements; ++i) {
        hostMatrixA[i] = static_cast<float>(i + 1.0f);         // e.g., 1.0, 2.0, ..., 12.0
        hostMatrixB[i] = static_cast<float>((i + 1.0f) * 0.5f); // e.g., 0.5, 1.0, ..., 6.0
    }

    // Print the initial matrices.
    printMatrix(hostMatrixA, rows, cols, "Matrix A (Host)");
    printMatrix(hostMatrixB, rows, cols, "Matrix B (Host)");

    // Call the CUDA function (defined in matrix_add.cu) to perform addition on the GPU.
    // .data() returns a pointer to the underlying array managed by std::vector.
    std::cout << "Performing matrix addition on the GPU..." << std::endl;
    addMatricesOnGPU(hostMatrixA.data(), hostMatrixB.data(), hostResultMatrixC.data(), rows, cols);
    std::cout << "Matrix addition on GPU complete." << std::endl << std::endl;

    // Print the result matrix C.
    printMatrix(hostResultMatrixC, rows, cols, "Result Matrix C (A+B) (Host)");

    // Optional: Demonstrate calls to functions from my_lib if needed.
    // std::cout << "Simple arithmetic from my_lib:" << std::endl;
    // std::cout << "add(10, 20) = " << add(10, 20) << std::endl; // Requires my_lib.h and linking my_lib

    return 0; // Indicate successful execution
}
