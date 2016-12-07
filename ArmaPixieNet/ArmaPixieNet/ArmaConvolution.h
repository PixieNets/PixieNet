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
    template<typename T> class ArmaConvolution;
}

template<typename T>
class aconv::ArmaConvolution {
private:
    uint            ac_size;                // kernel size (DIM 1, 2)
    uint            ac_channels;            // kernel channels (DIM 3)
    uint            ac_filters;             // kernel #filters (DIM 4)
    uint            ac_conv_stride;         // convolution stride
    uint            ac_conv_padding;        // convolution padding
    Convolution     ac_conv_type;           // convolution type
    Nonlinearity    ac_nl_type;             // non-linear activation function
    Pooling         ac_pool_type;           // pooling type
    uint            ac_pool_size;           // pooling size
    uint            ac_pool_stride;         // pooling stride
    arma::Mat<T>    ac_box_filter;          // convolution kernel to compute scaling factors of input
    arma::Cube<T>   *ac_conv_weights;       // 4D weights tensor for convolution
    T               *ac_alpha_per_filter;   // Scalar factors per weight filter

    // For the forward pass operations
    uint            n_in;
    uint            ac_ksz_half;
    uint            start_row;
    uint            end_row;
    uint            start_col;
    uint            end_col;
    uint            rows_out;
    uint            cols_out;
    uint            n_out;

    std::string     constructMessage(std::string functionName, std::string message);
    
public:
    ArmaConvolution(uint ksize, uint channels, uint filters, uint conv_stride, Convolution conv_type=Convolution::same,
                    Nonlinearity nl_type=Nonlinearity::relu, Pooling pool_type=Pooling::max,
                    uint pool_size=2, uint pool_stride=2);
    ~ArmaConvolution();
    
    // 1. Compute K matrix of input data (containing scalar factors per sub-tensor)
    void    getInputFactors(arma::Cube<T> *data, arma::Mat<T> &factors);
    // 2. Normalize input data by mean and variance (in-place)
    void    normalizeData3D(arma::Cube<T> *data, arma::Cube<T> &norm_input);
    // 3. Binarize and perform binary convolution
    void    convolve(arma::Cube<T> *data, const arma::Mat<T> &dataFactors, arma::Cube<T> *result);
    // 4. Non-linear activation (in-place)
    void    nlActivate(arma::Cube<T> *data);
    // 5. Pooling
    void    pool(arma::Cube<T> *input, arma::Mat<T> *result);
    // FULL forward pass of convolution unit in network
    void    forwardPass(arma::Cube<T> *input, arma::Cube<T> *result, arma::Cube<T> *result_pooling);     
    
    
    // Accessor functions for class members
    uint                size()              {   return ac_size;             }
    uint                channels()          {   return ac_channels;         }
    uint                filters()           {   return ac_filters;          }
    uint                conv_stride()       {   return ac_conv_stride;      }
    uint                conv_padding()      {   return ac_conv_padding;     }
    Convolution         conv_type()         {   return ac_conv_padding;     }
    Nonlinearity        nl_type()           {   return ac_nl_type;          }
    Pooling             pool_type()         {   return ac_pool_type;        }
    uint                pool_size()         {   return ac_pool_size;        }
    uint                pool_stride()       {   return ac_pool_stride;      }
    arma::Mat<T>        box_filter()        {   return ac_box_filter;       }
    arma::Cube<T>*      conv_weights()      {   return ac_conv_weights;     }
    T*                  alpha_per_filter()  {   return ac_alpha_per_filter; }
    
};