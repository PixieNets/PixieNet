//
//  ArmaConvolution.h
//  ArmaPixieNet
//
//  Created by Esha Uboweja on 12/7/16.
//  Copyright © 2016 Esha Uboweja. All rights reserved.
//

#pragma once

#include "armadillo"

typedef unsigned int uint;

namespace aconv {
    // Multiple types of pooling
    enum class Pooling {none, min, max, average};
    // Multiple types of convolution
    enum class Convolution {same, valid};
    // Multiple types of nonlinearities
    enum class Nonlinearity {none, relu};
    // Implementing XNOR convolution in Armadillo
    class ArmaConvolution;
}

template<typename T>
class aconv::ArmaConvolution {
private:
    uint ac_size;  // kernel size (DIM 1, 2)
    uint ac_channels;   // kernel channels (DIM 3)
    uint ac_filters;    // kernel #filters (DIM 4)
    uint ac_conv_stride;    // convolution stride
    uint ac_conv_padding;   // convolution padding
    Convolution ac_conv_type;  // convolution type
    Nonlinearity ac_nl_type;   // non-linear activation function
    Pooling ac_pool_type;   // pooling type
    uint ac_pool_size;  // pooling size
    uint ac_pool_stride;    // pooling stride
    arma::Mat<T> ac_box_filter; // convolution kernel to compute scaling factors of input
    std::vector<arma::Cube<T>> ac_conv_weights; // 4D weights tensor for convolution
    
    void init_convolution(uint ksize, uint channels, uint filters, uint conv_stride, Convolution conv_type);
    void init_pooling(Pooling pool_type=Pooling::none, uint pool_size=0, uint pool_stride=0);
    void init_nonlinearity(Nonlinearity actv_type=Nonlinearity::None);
    
public:
    ArmaConvolution(uint ksize, uint channels, uint filters, uint conv_stride, Convolution conv_type=Convolution::same,
                    Nonlinearity nl_type=Nonlinearity::relu, Pooling pool_type=Pooling::max,
                    uint pool_size=2, uint pool_stride=2);
    ~ArmaConvolution();
    
    // 1. Normalize input data by mean and variance
    arma::Cube<T>   normalizeData3D(const arma::Cube<T> &data);
    // 2. Compute K matrix of input data (containing scalar factors per sub-tensor)
    arma::Mat<T>    getScalars(const arma::Cube<T> &data);
    // 3.
    
    
    // Accessor functions for class members
    uint size() {   return ac_size; }
    uint channels() {   return ac_channels; }
    uint filters() {    return ac_filters; }
    uint conv_stride() {    return ac_conv_stride; }
    uint conv_padding() {   return ac_conv_padding;    }
    Convolution conv_type {    return ac_conv_padding;  }
    Pooling pool_type() {   return ac_pool_type;    }
    uint pool_size() {  return ac_pool_size;    }
    uint pool_stride() {    return ac_pool_stride;  }
    arma::Mat<T> box_filter() { return ac_box_filter;   }
    std::vector<arma::Cube<T>> conv_weights()   {   return ac_conv_weights; }
    
};