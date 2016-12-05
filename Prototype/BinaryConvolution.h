//
// Created by Esha Uboweja on 11/22/16.
//

#pragma once

#include "BinaryLayer.h"
#include "BinaryTensor3D.h"

using namespace bd;

// Multiple types of pooling
enum class Pooling {max, min, average};
// Multiple types of convolution
enum class Convolution {same, valid};

class BinaryConvolution {
private:
    uint               bc_width;
    uint               bc_height;
    uint               bc_channels;
    uint               bc_filters;
    uint               bc_conv_stride;
    uint               bc_padding;
    Convolution        bc_conv_type;
    bool               bc_pool;
    Pooling            bc_pool_type;
    uint               bc_pool_size;
    uint               bc_pool_stride;
    arma::mat          bc_box_filter;   // the kernel k applied to input A to get K
    BinaryTensor4D     bc_conv_weights; // Weights matrix for convolution

public:
    // Adding default values for pooling so that if pooling is set to false, user
    // doesn't have to provide pooling parameters
    BinaryConvolution(uint w, uint h, uint ch, uint k, uint stride, uint padding,
                      Convolution conv_type=Convolution::same, bool pool=true,
                      Pooling pool_type=Pooling::max, uint pool_size=2, uint pool_stride=2);
    ~BinaryConvolution();

    // 1. Normalize input data by mean and variance
    arma::mat      normalizeData2D(arma::mat  data);
    arma::cube     normalizeData3D(arma::cube data);
    // 2. Compute K matrix of input data
    arma::mat      input2KMat(arma::cube norm_data);
    // 3. Compute sign(I)
    BinaryTensor3D binarizeInput(arma::cube norm_data);
    // 4. Binary convolution
    arma::cube     doBinaryConv(BinaryTensor3D input, arma::mat K);
    // 5. Pooling
    arma::mat      poolMat(arma::mat data);
    arma::cube     doPooling(arma::cube data);
    // Setup pipeline - this is what a user would call
    arma::cube     forwardPass(arma::cube data);

    // Set weights
    void           setWeights(BinaryTensor4D conv_weights);

    // Accessor functions for class members
    uint width()              {    return bc_width;       }
    uint height()             {    return bc_height;      }
    uint channels()           {    return bc_channels;    }
    uint filters()            {    return bc_filters;     }
    uint conv_stride()        {    return bc_conv_stride; }
    uint padding()            {    return bc_padding;     }
    Convolution conv_type()   {    return bc_conv_type;   }
    bool pool()               {    return bc_pool;        }
    Pooling pool_type()       {    return bc_pool_type;   }
    uint pool_size()          {    return bc_pool_size;   }
    uint pool_stride()        {    return bc_pool_stride; }

};


