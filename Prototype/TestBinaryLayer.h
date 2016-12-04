//
// Created by Esha Uboweja on 12/4/16.
//

#pragma once

#include "BinaryLayer.h"

class TestBinaryLayer {
private:

    bool test_binarizeMat_single(uint rows = 3, uint cols = 3);
    bool test_binarizeMat();

    bool test_operatorMult_single(uint rows1 = 3, uint cols1 = 3, uint rows2 = 3, uint cols2 = 3);
    bool test_operatorMult_invalid(uint rows1, uint cols1, uint rows2, uint cols2);
    bool test_operatorMult();

    bool test_im2col_single(uint rows = 10, uint cols = 10,
                            uint block_width = 3, uint block_height = 3,
                            uint padding = 0, uint stride = 1);
    bool test_im2col_invalid(uint rows, uint cols,
                             uint block_width, uint block_height,
                             uint padding, uint stride);
    bool test_im2col();

    /*
    // These are similar to test_im2col
    bool test_repmat();
    bool test_reshape();
    */

    /*
    bool test_binarizeWeights();
    bool test_getDoubleWeights();
    */
public:
    bool runAllTests();
};
