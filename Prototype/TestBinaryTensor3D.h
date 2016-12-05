//
// Created by Esha Uboweja on 12/5/16.
//

#pragma once

#include "BinaryTensor3D.h"

class TestBinaryTensor3D {
private:
    bool test_im2col_single(uint rows = 10, uint cols = 10,
                            uint block_width = 3, uint block_height = 3,
                            uint padding = 0, uint stride = 1);
    bool test_im2col_invalid(uint rows, uint cols,
                             uint block_width, uint block_height,
                             uint padding, uint stride);
    bool test_im2col();

public:
    void runAllTests();
};

