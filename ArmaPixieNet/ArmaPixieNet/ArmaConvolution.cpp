//
//  ArmaConvolution.cpp
//  ArmaPixieNet
//
//  Created by Esha Uboweja on 12/7/16.
//  Copyright © 2016 Esha Uboweja. All rights reserved.
//

#include <stdio.h>
#include "ArmaConvolution.h"

using namespace aconv;

/*
 * Initializes a convolutional layer
 * @param ksize - square size of convolution kernel (DIMS 1,2)
 * @param channels - #channels (DIM 3) of the weights 4D hypercube
 * @param filters - #filters in the weights 4D hypercube (DIM 4)
 * @param conv_stride - stride for convolution
 * @param conv_type - convolution type, one of "same" or "valid"
 * @param nl_type - type of non-linear activation, one of "none" or "relu"
 * @param pool_type - type of pooling, one of "none", max", "min", or "average"
 * @param pool_size - size of square pooling kernel
 * @param pool_stride - stride for pooling operations
 */
template<typename T>
ArmaConvolution<T>::ArmaConvolution(uint ksize, uint channels, uint filters, uint conv_stride, 
									Convolution conv_type, Nonlinearity nl_type, Pooling pool_type,
                    				uint pool_size, uint pool_stride) {
	// Convolution
	this->ac_size = ksize;
	this->ac_channels = channels;
	this->ac_filters = filters;
	this->ac_conv_stride = conv_stride;
	this->ac_conv_type = conv_type;
	if (this->ac_conv_type == Convolution::same) {
		this->ac_conv_padding = this->ac_size / 2;
	} else if (this->ac_conv_type == Convolution::valid) {
		this->ac_conv_padding = 0;
	}
	// Non-linearity
	this->ac_nl_type = nl_type;
	// Pooling
	this->ac_pool_type = pool_type;
	this->ac_pool_size = pool_size;
	this->ac_pool_stride = pool_stride;
	// Normalization
	this->ac_box_filter = arma::ones<arma::Mat<T>>(ksize * ksize) * (1.0 / (ksize * ksize));
	// Convolution weights, 4-D hypercube
    this->ac_conv_weights = new arma::Cube<T>[filters];
    // Convolution weight alpha values per filter
    this->ac_alpha_per_filter = new T[filters];
}

template<typename T>
ArmaConvolution<T>::~ArmaConvolution() {
	// delete the 4D weights hypercube array
	delete[] this->ac_conv_weights;
	// delete the alpha values array
	delete[] this->ac_conv_weights;
}


