//
// Created by Zal on 11/22/16.
//

#include "TestArmadillo.h"
#include <algorithm> // for std::sort

void TestArmadillo::testSoftmax() {
    std::cout << "--- TEST SOFTMAX" << std::endl;
    // The test will have 10 output from previous layers
    // and 5 possible classes as an output
    // the activation function is a sigmoid
    int numOutput = 8;
    int numLastOutput = 12;

    std::cout << "SUM [1] :" << std::endl;
    vec S = ones<vec>(numLastOutput);
    std::cout << accu(S) << std::endl;

    std::cout << "EXP^[1]:" << std::endl;
    std::cout << exp(S) << std::endl;

    std::cout << "DIV [1]/2:" << std::endl;
    std::cout << S/2.0 << std::endl;

    std::cout << "SOFTMAX:" << std::endl;
    mat W = ones<mat>(numLastOutput,numOutput);
    mat lastOutput = ones<vec>(numLastOutput);
    vec Y = exp(tanh(W.t() * lastOutput));
    double normC = accu(Y);
    std::cout << "Expected value: " << 1/double(numOutput) << std::endl;
    std::cout << Y/normC << std::endl;
}

void TestArmadillo::testRank() {
    std::cout << "--- TEST RANK VECTOR" << std::endl;


    arma_rng::set_seed(1234);
    arma::vec randVec = arma::randn(10)*10;
    std::cout << "randVec:" << std::endl;
    std::cout << randVec << std::endl;

    arma::uvec sortedIdxVec = arma::sort_index(randVec, "descend");
    std::cout << "sorted vec idx:" << std::endl;
    std::cout << sortedIdxVec << std::endl;

    for(int i=0; i < 5; ++i) {
        std::cout << randVec.at(sortedIdxVec(i)) << std::endl;
    }

}

void TestArmadillo::testGeneral() {
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
}

void TestArmadillo::testFlattenCube(){
    std::cout << "--- TEST FLATTEN CUBE" << std::endl;
    //filter's dimensions are defined as [rows, cols, numFilters]
    arma::cube randCube = randu<arma::cube>(1, 1, 10);
    std::cout << "Random cube" << std::endl;
    std::cout << randCube << std::endl;

    randCube.reshape(10,1,1);
    std::cout << "Vectorized cube" << std::endl;
    std::cout << randCube << std::endl;
}

void TestArmadillo::runTest() {
    this->testGeneral();
    this->testSoftmax();
    this->testRank();
    this->testFlattenCube();
}