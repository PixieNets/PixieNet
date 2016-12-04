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
    bool test_operatorMult();
    /*
    bool test_binarizeWeights();
    bool test_getDoubleWeights();
    bool test_operatorMult();
    bool test_im2col();
    bool test_repmat();
    bool test_reshape();
    */
public:
    bool runAllTests();
};
