//
// Created by Esha Uboweja on 11/22/16.
//

#pragma once

#include "BinaryLayer.h"

// Multiple types of pooling
enum class Pooling {max, min, average};

// Concatenate multiple binary layer channels to form a 3D binary tensor
typedef BinaryLayer** BinaryTensor;

class BinaryConvolution {
private:
    uint         bc_width;
    uint         bc_height;
    uint         bc_channels;
    uint         bc_stride;
    bool         bc_pool;
    Pooling      bc_pool_type;
    uint         bc_pool_stride_width;
    uint         bc_pool_stride_height;
    arma::mat    bc_box_filter;   // the kernel k applied to input A to get K
    BinaryTensor bc_conv_weights; // Weights matrix for convolution

public:
    // Adding default values for pooling so that if pooling is set to false, user
    // doesn't have to provide pooling parameters
    BinaryConvolution(uint w, uint h, uint ch, uint stride,
                      bool pool=true, Pooling pool_type=Pooling::max,
                      uint pool_stride_width=2, uint pool_stride_height=2);
    ~BinaryConvolution();

    // 1. Normalize input data by mean and variance
    arma::mat    normalizeData2D(arma::mat  data);
    arma::cube   normalizeData3D(arma::cube data);
    // 2. Compute K matrix of input data
    arma::mat    input2KMat(arma::cube norm_data);
    // 3. Compute sign(I)
    BinaryTensor binarizeInput(arma::cube norm_data);
    // 4. Binary convolution
    arma::cube   doBinaryConv(BinaryTensor input, arma::mat K);
    // 5. Pooling
    arma::mat    poolMat(arma::mat data);
    arma::cube   doPooling(arma::cube data);

    // Accessor functions for class members
    uint width()              {    return bc_width;    }
    uint height()             {    return bc_height;   }
    uint channels()           {    return bc_channels; }
    uint stride()             {    return bc_stride;   }
    bool pool()               {    return bc_pool;     }
    Pooling pool_type()       {    return bc_pool_type; }
    uint pool_stride_width()  {    return bc_pool_stride_width; }
    uint pool_stride_height() {    return bc_pool_stride_height; }

};


