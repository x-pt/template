#include "my_lib.h"

int add(const int a, const int b) {
    return a + b;
}

int sub(const int a, const int b) {
    return a - b;
}

int mul(const int a, const int b) {
    return a * b;
}

double divide(const double a, const double b) {
    if (b == 0.0) {
        throw std::invalid_argument("Division by zero");
    }
    return a / b;
}
