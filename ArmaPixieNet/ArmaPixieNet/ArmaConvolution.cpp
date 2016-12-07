//
//  ArmaConvolution.cpp
//  ArmaPixieNet
//
//  Created by Esha Uboweja on 12/7/16.
//  Copyright © 2016 Esha Uboweja. All rights reserved.
//

#include <stdio.h>
#include <stdexcept>
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
	this->ac_ksz_half = ksize/2;
	this->start_row = 0;
	this->start_col = 0;
	this->end_row = 0;
	this->end_col = 0;

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

template<typename T>
std::string ArmaConvolution<T>::constructMessage(std::string functionName, std::string message) {
	return std::string("[ArmaConvolution::") + functionName + std::string("] ") + message;
}

template<typename T>
void ArmaConvolution<T>::forwardPass(arma::Cube<T> *input, arma::Cube<T> *result,
								 	arma::Cube<T> *result_pooling) {

	std::string fname = "forwardPass";
	if (input->empty()) {
		throw std::invalid_argument(this->constructMessage(fname, "Input data must be non-empty"));
	}
	if (input->n_slices != this->n_channels) {
		std::string message = std::string("Input data #channels = ") + std::to_string(input->n_slices)
							+ std::string(" should match convolution weights 4D hypercube #channels = ")
							+ std::to_string(this->n_channels);
		throw std::invalid_argument(this->constructMessage(fname, message));
	}
	if (!result->empty()) {
		throw std::invalid_argument(this->constructMessage(fname, 
									"Output 3D hypercube must be empty to fill in with the correct dims"));
	}


	// Initialize output dimensions
	this->rows_out = input->n_rows - this->ac_size + 2 * this->padding;
	this->cols_out = input->n_cols - this->ac_size + 2 * this->padding;
	if (this->rows_out % this->ac_conv_stride || this->cols_out % this->ac_conv_stride) {
		std::string message = std::string("Input data dimensions (") + std::to_string(input->nrows) + 
							  std::string(", ") + std::to_string(input->n_cols) + 
							  std::string(") are invalid for convolution with weights of size (") + 
							  std::to_string(this->ac_size) + std::string(", ") + std::to_string(this->ac_size)
							  + std::string(", ") + std::to_string(this->ac_channels) + std::string(", ")
							  + std::to_string(this->ac_filters) + std::string(") with stride = ") + 
							  std::to_string(this->ac_conv_stride) + std::string(" and padding = ") +
							  std::to_string(this->ac_conv_padding);
		throw std::invalid_argument(this->constructMessage(fname, message));
	}
	this->rows_out = this->rows_out / this->ac_conv_stride + (this->rows_out % 2);
	this->cols_out = this->cols_out / this->ac_conv_stride + (this->cols_out % 2);

	// Initialize traversal values
	this->start_row = 0; this->start_col = 0;
	this->end_row = input->n_rows; this->end_col = input->n_cols;
	if (this->ac_conv_padding == 0) {
		start_row += this->ac_ksz_half;
		start_col += this->ac_ksz_half;
		end_row -= this->ac_ksz_half;
		end_col -= this->ac_ksz_half;
	}

	// 1. Compute K matrix of input data (containing scalar factors per sub-tensor)
	arma::Mat<T> *inputFactors;
	this->getInputFactors(input, inputFactors);
	// 2. Normalize input data by mean and variance (in-place)
    this->normalizeData3D(input);
    // 3. Binarize and perform binary convolution
    this->convolve(input, inputFactors, result);
    // 4. Non-linear activation (in-place)
    if (this->ac_nl_type != Nonlinearity::none) {
    	this->nlActivate(result);
    }
    // 5. Pooling
    if (this->ac_pool_type != Pooling::none) {
    	this->pool(result, result_pooling);
    }
    // Done!
}












