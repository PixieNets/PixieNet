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

bool TestBinaryLayer::test_operatorMult_single(uint rows1, uint cols1, uint rows2, uint cols2) {
     // Generate 2 random binary matrices
    arma::umat input2D_a = BinaryMatrix::randomArmaUMat(rows1, cols1);
    BinaryLayer bl_a(input2D_a);
    arma::umat input2D_b = BinaryMatrix::randomArmaUMat(rows2, cols2);
    BinaryLayer bl_b(input2D_b);

#ifdef DEBUG
    std::cout << "[test_operatorMult_single] arma input 1: \n" << input2D_a << std::endl;
    std::cout << "[test_operatorMult_single] binary matrix input 1: \n";
    bl_a.binMtx()->print();
    std::cout << std::endl;
    std::cout << "[test_operatorMult_single] arma input 2: \n" << input2D_b << std::endl;
    std::cout << "[test_operatorMult_single] binary matrix input 2: \n";
    bl_b.binMtx()->print();
    std::cout << std::endl;
#endif

    // ARMA product
    arma::imat ia = arma::conv_to<imat>::from(input2D_a);
    ia.replace(0, -1);
    arma::imat ib = arma::conv_to<imat>::from(input2D_b);
    ib.replace(0, -1);
    arma::imat ires = ia % ib;
    ires.replace(-1, 0);
    arma::umat armaResult = arma::conv_to<umat>::from(ires);

    // XNOR product
    BinaryLayer result = bl_a * bl_b;

#ifdef DEBUG
    ires.replace(0, -1);
    std::cout << "[test_operatorMult_single] Arma integer result: \n" << ires << std::endl;
    std::cout << "[test_operatorMult_single] Arma result: \n" << armaResult << std::endl;
    std::cout << "[test_operatorMult_single] Binary layer result: \n";
    result.binMtx()->print();
    std::cout << std::endl;
#endif

    return result.binMtx()->equalsArmaMat(armaResult);
}

bool TestBinaryLayer::test_operatorMult_invalid(uint rows1, uint cols1, uint rows2, uint cols2) {
    try {
        test_operatorMult_single(rows1, cols1, rows2, cols2);
    } catch (std::exception e) {
        return true;
    }
    // didn't raise exception
    return false;
}

bool TestBinaryLayer::test_operatorMult() {
    return test_operatorMult_single()
        && test_operatorMult_single(5, 6, 5, 6)
        && test_operatorMult_invalid(5, 6, 7, 8);
}

bool TestBinaryLayer::test_im2col_single(uint rows, uint cols, uint block_width, uint block_height, uint padding,
                                         uint stride) {
    // Generate a random binary matrix
    arma::umat input2D = BinaryMatrix::randomArmaUMat(rows, cols);
    BinaryLayer bl(input2D);

#ifdef DEBUG
    std::cout << "[test_im2col_single] arma input:\n" << input2D << std::endl;
    std::cout << "[test_im2col_single] bm input:\n" ;
    bl.binMtx()->print();
    std::cout << std::endl;
#endif

    // Compare im2col result for binary matrix and arma
    arma::umat armaResult = BinaryMatrix::im2colArmaMat(input2D, block_width, block_height, padding, stride);
    BinaryLayer blResult = bl.im2col(block_width, block_height, padding, stride);

#ifdef DEBUG
    std::cout << "[test_im2col_single] arma result:\n" << armaResult << std::endl;
    std::cout << "[test_im2col_single] bm result:\n" ;
    blResult.binMtx()->print();
    std::cout << std::endl;
#endif

    return blResult.binMtx()->equalsArmaMat(armaResult);
}

bool TestBinaryLayer::test_im2col_invalid(uint rows, uint cols, uint block_width, uint block_height, uint padding, uint stride) {
    try {
        test_im2col_single(rows, cols, block_width, block_height, padding, stride);
    } catch (std::exception e) {
        return true;
    }
    std::cerr << "[test_im2col_invalid] Test didn't raise exception\n";
    // This test should raise an exception
    return false;
}


bool TestBinaryLayer::test_im2col() {
    return test_im2col_single()
        && test_im2col_single(3, 3)
        && test_im2col_single(3, 3, 3, 3, 1, 1)
        && test_im2col_single(3, 3, 3, 3, 1, 2)
        && test_im2col_single(5, 5, 3, 3, 1, 2)
        && test_im2col_single(7, 9, 5, 5, 2, 1)
        && test_im2col_invalid(8, 6, 3, 3, 1, 2)
        && test_im2col_invalid(7, 9, 5, 5, 3, 1)
        && test_im2col_invalid(10, 10, 3, 3, 0, 2);
}

bool TestBinaryLayer::runAllTests() {
    return test_binarizeMat()
        && test_operatorMult()
        && test_im2col();
}