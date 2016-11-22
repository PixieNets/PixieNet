//
// Created by Esha Uboweja on 11/22/16.
//

#pragma once

#include "BinaryLayer.h"

// Concatenate multiple binary layer channesl to form a 3D binary tensor
typedef BinaryLayer** BinaryTensor;

class BinaryConvolution {
private:
    int          bc_width;
    int          bc_height;
    int          bc_channels;
    bool         bc_pool;
    BinaryTensor bc_conv_weights; // Weights matrix for convolution

public:
    BinaryConvolution(int w, int h, int ch, bool pool);
    ~BinaryConvolution();

    // Accessor functions for class members
    int width()    {    return bc_width;    }
    int height()   {    return bc_height;   }
    int channels() {    return bc_channels; }
    bool pool()    {    return bc_pool;     }
};


