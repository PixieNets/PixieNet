#include <iostream>
#include <armadillo>

#include "TestBinaryMatrix.h"

using namespace arma;

int main() {

    // Test armadillo loading
    mat A = randu<mat>(4,5);
    mat B = randu<mat>(4,5);

    std::cout << "Multiplying two random 4 x 5 matrices using Armadillo:\n";
    std::cout << A * B.t() << std::endl;

    TestBinaryMatrix test;
    test.runAllTests();

    return 0;
}