#include <iostream>
#include <armadillo>

#include "TestBinaryMatrix.h"

using namespace arma;

int main() {

    // Test armadillo loading
    mat A = randu<mat>(4,5);
    mat B = randu<mat>(4,5);
    cube C = randu<cube>(4,5,3);
    //cube D = zeros<cube>(4, 5, 3);
    cube D = zeros<cube>(size(C)); // WORKS!
    mat E = ones<mat>(3, 3) * (1.0 / (3 * 3)); // WORKS!
    mat im = randu<mat>(5, 5);
    mat box_res = conv2(im, E, "same");
    mat sub_mat = E(span(0,1), span(0, 1));
    sub_mat(0, 1) = -1;
    sub_mat(1, 0) = -1;
    mat zeros_mat = arma::zeros<mat>(size(sub_mat));
    mat relu_sub_mat = sub_mat;
    relu_sub_mat.elem( find(relu_sub_mat < 0)).zeros();

    std::cout << "sub_mat: \n" << sub_mat << std::endl;
    std::cout << "relu(sub_mat): \n" << arma::max(zeros_mat, sub_mat) << std::endl;
    std::cout << "relu(sub_mat): \n" << relu_sub_mat << std::endl;
    std::cout << "E: \n" << E << std::endl;
    std::cout << "im: \n" << im << std::endl;
    std::cout << "box filtered result: \n" << box_res << std::endl;
    std::cout << "D: \n" << D << std::endl;
    std::cout << "A[0] = " << A[0] << std::endl;
    std::cout << "sum(sum(A)): " << sum(sum(A)) << std::endl;
    std::cout << "Multiplying two random 4 x 5 matrices using Armadillo:\n";
    std::cout << A * B.t() << std::endl;

    TestBinaryMatrix test;
    test.runAllTests();

    return 0;
}