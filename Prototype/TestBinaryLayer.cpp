//
// Created by Esha Uboweja on 12/4/16.
//

#include <armadillo>
#include "BinaryLayer.h"
#include "TestBinaryLayer.h"

#define DEBUG 0

using namespace bd;

bool TestBinaryLayer::test_binarizeMat_single(uint rows, uint cols) {
    arma::mat values = arma::randn(rows, cols);

    BinaryLayer bl = BinaryLayer(cols, rows);

#ifdef DEBUG
    std::cout << "[test_binarizeMat_single] arma input:\n" << values << std::endl;
#endif

    // Compare Armadillo result and BinaryMatrix result
    double armaAlpha = arma::mean(arma::mean(arma::abs(values)));
    arma::umat armaResult(rows, cols);
    armaResult.zeros();
    armaResult.elem(arma::find(values >= 0)).ones();
    armaResult.elem(arma::find(values < 0)).zeros();
    bl.binarizeMat(values);

#ifdef DEBUG
    std::cout << "[test_binarizeMat_single] Arma alpha: " << armaAlpha << " Arma output:\n" << armaResult << std::endl;
    std::cout << "[test_binarizeMat_single] Binarized alpha: " << bl.alpha() << " output (bl):\n";
    bl.binMtx()->print();
    std::cout << std::endl;
#endif

    return (bl.alpha() == armaAlpha) && bl.binMtx()->equalsArmaMat(armaResult);
}

bool TestBinaryLayer::test_binarizeMat() {
    return test_binarizeMat_single()
        && test_binarizeMat_single(5, 3)
        && test_binarizeMat_single(28, 9)
        && test_binarizeMat_single(81, 102);
}

bool TestBinaryLayer::runAllTests() {
    return test_binarizeMat();
}