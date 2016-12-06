//
// Created by Zal on 11/20/16.
//

#pragma once

#include <iostream>
#include "BinaryMatrix.h"

class TestBinaryMatrix {
    // Generate
    BinaryMatrix generateDiag(uint n);
    BinaryMatrix generateUpperDiag(uint n);
    BinaryMatrix generateLowerDiag(uint n);

    // Tests
    void testCreateAndPrint();
    void testGetBit();
    void testSetBit();
    void testSetValueAt();
    void testTransposeIdx();
    void testTranspose();
    void testBinMultiply();
    void testTBinMultiply();
    void testDoubleMultiply();


    void printTestUArma2(std::string testName, std::string desc, arma::umat input);
    void printTestArma2(std::string testName, std::string desc, arma::mat input);
    void printTestBM(std::string testName, std::string desc, BinaryMatrix input);

    bool test_setValueAtIdx_single(uint rows = 3, uint cols = 5, uint8 value=BIT_ONE);
    bool test_initWithArma_single(uint rows = 10, uint cols = 10);
    bool test_im2col_single(uint rows = 10, uint cols = 10,
                            uint block_width = 3, uint block_height = 3,
                            uint padding = 0, uint stride = 1);
    bool test_im2col_invalid(uint rows, uint cols,
                             uint block_width, uint block_height,
                             uint padding, uint stride);
    bool test_repmat_single(uint rows = 2, uint cols = 3, uint n_rows = 2, uint n_cols = 3);
    bool test_reshape_single(uint rows = 2, uint cols = 3, uint new_rows = 3, uint new_cols = 2);
    bool test_reshape_invalid(uint rows = 2, uint cols = 3, uint new_rows = 3, uint new_cols = 2);
    bool test_bitCountPerRow_single(uint rows = 10, uint cols = 100, bool reshape = false, uint new_rows = 0, uint new_cols = 0);
    bool test_bitCountPerCol_single(uint rows = 10, uint cols = 100, bool reshape = false, uint new_rows = 0, uint new_cols = 0);

    bool test_setValueAtIdx();
    bool test_initWithArma();
    bool test_im2col();
    bool test_repmat();
    bool test_reshape();
    bool test_bitCountPerRow();
    bool test_bitCountPerCol();

public:
    void runAllTests();
};

