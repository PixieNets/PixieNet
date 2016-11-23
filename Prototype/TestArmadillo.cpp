//
// Created by Zal on 11/22/16.
//

#include "TestArmadillo.h"

void TestArmadillo::runTest() {
    // Test armadillo loading
    mat A = randu<mat>(4,5);
    mat B = randu<mat>(4,5);

    std::cout << "----- TEST TRANSPOSE INDEX" << std::endl;
    std::cout << "A[0] = " << A[0] << std::endl;
    std::cout << "sum(sum(A)): " << sum(sum(A)) << std::endl;
    std::cout << "Multiplying two random 4 x 5 matrices using Armadillo:\n";
    std::cout << A * B.t() << std::endl;
}