//
// Created by Esha Uboweja on 11/22/16.
//

#pragma once

#include "BinaryLayer.h"
#include "BinaryTensor3D.h"

using namespace bd;

typedef std::vector<arma::ucube> ArmaUTensor4D;

namespace bconv {
    // Multiple types of pooling
    enum class Pooling {none, max, min, average};
    // Multiple types of convolution
    enum class Convolution {same, valid};
    // Multiple types of non-linearities
    enum class Nonlinearity {none, relu};
    class BinaryConvolution;
};

class bconv::BinaryConvolution {
private:
    uint               bc_width;
    uint               bc_height;
    uint               bc_channels;
    uint               bc_filters;
    uint               bc_conv_stride;
    uint               bc_padding;
    Convolution        bc_conv_type;
    bool               bc_nonlinear_actv;
    Nonlinearity       bc_nonlinearity;
    bool               bc_pool;
    Pooling            bc_pool_type;
    uint               bc_pool_size;
    uint               bc_pool_stride;
    arma::mat          bc_box_filter;   // the kernel k applied to input A to get K
    BinaryTensor4D     bc_conv_weights; // Weights matrix for convolution

    void init_convolution(uint w, uint h, uint ch, uint k, uint stride, Convolution conv_type);
    void init_pooling(Pooling pool_type=Pooling::none, uint pool_size=0, uint pool_stride=0);
    void init_nonlinearity(Nonlinearity actv_type=Nonlinearity::none);

public:
    // Adding default values for pooling so that if pooling is set to false, user
    // doesn't have to provide pooling parameters
    // Note: the kernel is of different width and height but presently works for square padding
    BinaryConvolution(uint w, uint h, uint ch, uint k, uint stride,
                      Convolution conv_type=Convolution::same,
                      Nonlinearity actv_type=Nonlinearity::relu,
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
    // 5. Non-linearity
    arma::cube     nonLinearActivate(arma::cube data);
    // 6. Pooling
    arma::mat      poolMat(arma::mat data);
    arma::cube     doPooling(arma::cube data);
    // Setup pipeline - this is what a user would call
    arma::cube     forwardPass(arma::cube data);
    arma::cube     forwardPassArma(arma::cube data);


    static BinaryLayer          bt4_reshape(BinaryTensor4D tensor, uint new_width, uint new_height);

    // Compute standard deviation of all elements of a 2D Arma matrix
    static double               std2Arma(arma::mat input);
    // Get a random 4D binary matrix
    static ArmaUTensor4D        randomTensor4DUArma(uint width, uint height, uint channels, uint filters);
    static BinaryTensor4D       randomTensor4D(uint width, uint height, uint channels, uint filters, uint nrandom = 0);
    // String representaiton of a 4D Binary Tensor
    static std::string          bt4ToString(BinaryTensor4D input);
    // Convert a 4D Arma tensor to a 4D Binary Tensor
    static BinaryTensor4D       uarmaToBT4(ArmaUTensor4D input);
    // Binary convolution for Arma tensor weights
    static arma::cube           armaBinaryConv(arma::ucube input, arma::mat K, ArmaUTensor4D weights, uint stride,
                                               Convolution conv_type, std::vector<double> alphaPerFilter);

    // Set weights
    void           setWeights(BinaryTensor4D conv_weights);
    void           setPadding(uint padding) {   this->bc_padding = padding;     }
    void           setStride(uint stride)   {   this->bc_conv_stride = stride;  }

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


