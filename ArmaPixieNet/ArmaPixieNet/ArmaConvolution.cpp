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
 * @param pool_type - type of pooling, one of "none", or max"
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
		throw std::invalid_argument(constructMessage(fname, "Input data must be non-empty"));
	}
	if (input->n_slices != this->n_channels) {
		std::string message = std::string("Input data #channels = ") + std::to_string(input->n_slices)
							+ std::string(" should match convolution weights 4D hypercube #channels = ")
							+ std::to_string(this->n_channels);
		throw std::invalid_argument(constructMessage(fname, message));
	}
	if (!result->empty()) {
		throw std::invalid_argument(constructMessage(fname, 
									"Output 3D hypercube must be empty to fill in with the correct dims"));
	}

	// Initialize output dimensions
	this->n_in = input->n_rows * input->n_cols;
	this->rows_out = input->n_rows - this->ac_size + 2 * this->padding;
	this->cols_out = input->n_cols - this->ac_size + 2 * this->padding;
	if (this->rows_out % this->ac_conv_stride || this->cols_out % this->ac_conv_stride) {
		std::string message = std::string("Input data dimensions (") + std::to_string(input->n_rows) + 
							  std::string(", ") + std::to_string(input->n_cols) + 
							  std::string(") are invalid for convolution with weights of size (") + 
							  std::to_string(this->ac_size) + std::string(", ") + std::to_string(this->ac_size)
							  + std::string(", ") + std::to_string(this->ac_channels) + std::string(", ")
							  + std::to_string(this->ac_filters) + std::string(") with stride = ") + 
							  std::to_string(this->ac_conv_stride) + std::string(" and padding = ") +
							  std::to_string(this->ac_conv_padding);
		throw std::invalid_argument(constructMessage(fname, message));
	}
	this->rows_out = this->rows_out / this->ac_conv_stride + (this->rows_out % 2);
	this->cols_out = this->cols_out / this->ac_conv_stride + (this->cols_out % 2);
	this->n_out = this->rows_out * this->cols_out;

	// Initialize traversal values
	this->start_row = 0; this->start_col = 0;
	this->end_row = input->n_rows; this->end_col = input->n_cols;
	if (this->ac_conv_padding == 0) {
		this->start_row += this->ac_ksz_half;
		this->start_col += this->ac_ksz_half;
		this->end_row -= this->ac_ksz_half;
		this->end_col -= this->ac_ksz_half;
	}

	// 1. Compute K matrix of input data (containing scalar factors per sub-tensor)
	arma::Mat<T> input_factors;
	this->getInputFactors(input, input_factors);
	// 2. Normalize input data by mean and variance (in-place)
	arma::Cube<T> norm_input;
    this->normalizeData3D(input, norm_input);
    // 3. Binarize and perform binary convolution
    // this->result = new Arma::Cube<T>(this->rows_out, this->cols_out, this->ac_filters);
    this->convolve(norm_input, input_factors, result);
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

// 1. Compute K matrix of input data (containing scalar factors per sub-tensor)
template<typename T>
void ArmaConvolution<T>::getInputFactors(arma::Cube<T> *data, arma::Mat<T> &factors) {
	// A = (\sum_{i=1}^n I(:, :, i)) / n
	arma::mat A = arma::mean(*data, 2);
	// Convolve with box filter k of size w x h of the convolution weights
	factors = arma::conv2(A, this->ac_box_filter, "same");
	if (this->ac_conv_stride > 1) {
		uint i = 0;
		arma::uvec indices(this->n_out);
		for (uint col = this->start_col; col < this->end_col; ++col) {
			for (uint row = this->start_row; row < this->end_row; ++row) {
				indices(i++) = arma::sub2ind(arma::size(factors), row, col);
			}
		}
		factors = arma::reshape(factors.elem(indices), this->rows_out, this->cols_out);
	}
}

// 2. Normalize input data by mean and variance (in-place)
template<typename T>
void ArmaConvolution<T>::normalizeData3D(arma::Cube<T> *data, arma::Cube<T> &norm_input) {
	norm_input = arma::zeros(arma::size(data));
	for (uint ch = 0; ch < this->ac_channels; ++ch) {
		T mean_value = arma::accu(data->slice(ch)) / this->n_in;
		arma::mat elems = ((data->slice(ch) - mean_value) % (data->slice(ch) - mean_value))
							(this->n_in - 1.0);
		norm_input.slice(ch) = arma::sqrt(elems);
	}
}

// 3. Binarize and perform binary convolution
template<typename T>
void ArmaConvolution<T>::convolve(const arma::Cube<T> &data, const arma::Mat<T> &dataFactors, 
								  arma::Cube<T> *result) {
	// 1. Binarize input
	arma::Cube<T> bin_input = arma::sign(data);
	bin_input.replace(0, 1);

	// 2. Perform convolution

}

// 4. Non-linear activation (in-place)
template<typename T>
void ArmaConvolution<T>::nlActivate(arma::Cube<T> *data) {
	std::string fname = "nlActivate";
	if (this->ac_nonlinearity == Nonlinearity::relu) {
		data->elem(arma::find(*data < 0)).zeros();
	} else {
		throw std::invalid_argument(constructMessage(fname, 
										"Unidentified Non-linearity function"));
	}
}

// 5. Pooling
template<typename T>
void ArmaConvolution<T>::pool(arma::Cube<T> *input, arma::Cube<T> *result) {
	std::string fname = "pool";
	if (input->empty()) {
		throw std::invalid_argument(constructMessage(fname, "Input should be non-empty"));
	}
	if (this->ac_pool_type != Pooling::max) {
		throw std::invalid_argument(constructMessage(fname, "Unidentified pooling type"));
	}
	uint new_rows = (uint) (input->n_rows - this->ac_pool_size) / this->ac_pool_stride - 1;
	uint new_cols = (uint) (input->n_cols - this->ac_pool_size) / this->ac_pool_stride - 1;

	result = new arma::Cube<T>(new_rows, new_cols, input->channels);

	uint srow = 0, scol = 0, erow = 0, ecol = 0;
	for (uint row = 0; row < this->rows_out; ++row) {
		for (uint col = 0; col < this->cols_out; ++col) {
			srow = row * this->ac_pool_size;
			erow = srow + this->ac_pool_size - 1;
			scol = col * this->ac_pool_size;
			ecol = scol + this->ac_pool_size - 1;
			result->at(row, col) = arma::max(arma::max(input->at(arma::span(srow, erow), 
																 arma::span(scol, ecol))));
		}
	}
}

// Convolution parameters
template<typename T>
void ArmaConvolution<T>::set_conv_params(uint rows_in, uint cols_in, uint ksize, 
										uint stride, uint padding, aconv_params *params) {
	std::string fname = "set_conv_params";
	if (params == NULL) {
		throw std::invalid_argument(constructMessage(fname, "params struct is NULL"));
	}
	params->ksize = ksize;
	params->n_in = rows_in * cols_in;
	params->rows_out = rows_in - ksize + 2 * padding;
	params->cols_out = cols_in - ksize + 2 * padding;
	if (params->rows_out % stride || params->cols_out % stride) {
		std::string message = std::string("Input data dimensions (") + std::to_string(rows_in) + 
							  std::string(", ") + std::to_string(colsin) + 
							  std::string(") are invalid for convolution with weights of size (") + 
							  std::to_string(ksize) + std::string(", ") + std::to_string(ksize) +
							  std::string(") with stride = ") + 
							  std::to_string(stride) + std::string(" and padding = ") +
							  std::to_string(padding);
		throw std::invalid_argument(constructMessage(fname, message));
	}
	params->rows_out = params->rows_out / stride + (params->rows_out % 2);
	params->cols_out = params->cols_out / stride + (params->cols_out % 2);
	params->n_out = params->rows_out * params->cols_out;
	params->ksz_half = ksize/2;
	params->start_row = 0;
	params->start_col = 0;
	params->end_row = rows_in;
	params->end_col = cols_in;
	if (padding == 0) {
		params->start_row = params->ksz_half;
		params->end_row -= params->ksz_half;
		params->start_col = params->ksz_half;
		params->end_col -= params->ksz_half;
	}
}

// Convolution:
// Input - 2D, Weights - 2D
template<typename T>
void ArmaConvolution<T>::convolve2D(arma::Mat<T> *input, arma::Mat<T> *weights,
								  	uint stride, uint padding, arma::Mat<T> *result) {
	std::string fname = "convolve2D";
	if (!input || input->empty()) {
		throw std::invalid_argument(constructMessage(fname, "Input should be non-empty"));
	}
	if (!weights || weights->empty()) {
		throw std::invalid_argument(constructMessage(fname, "Weight kernels should be non-empty"));
	}
	if (weights->n_rows != weights->n_cols) {
		throw std::invalid_argument(constructMessage(fname, "Weight kernels should be square size"));
	}

	// Initialize output dimensions
	uint ksize = weights->n_rows;
	uint n_in = input->n_rows * input->n_cols;
	uint rows_out = input->n_rows - ksize + 2 * padding;
	uint cols_out = input->n_cols - ksize + 2 * padding;
	if (rows_out % stride || cols_out % stride) {
		std::string message = std::string("Input data dimensions (") + std::to_string(input->n_rows) + 
							  std::string(", ") + std::to_string(input->n_cols) + 
							  std::string(") are invalid for convolution with weights of size (") + 
							  std::to_string(ksize) + std::string(", ") + std::to_string(ksize) +
							  std::string(") with stride = ") + 
							  std::to_string(stride) + std::string(" and padding = ") +
							  std::to_string(padding);
		throw std::invalid_argument(constructMessage(fname, message));
	}
	rows_out = rows_out / stride + (rows_out % 2);
	cols_out = cols_out / stride + (cols_out % 2);
	n_out = rows_out * cols_out;

	// Initialize traversal values
	uint ksz_half = ksize / 2;
	uint start_row = 0, start_col = 0;
	uint end_row = input->n_rows, end_col = input->n_cols;
	if (padding == 0) {
		start_row += ksz_half;
		start_col += ksz_half;
		end_row -= ksz_half;
		end_col -= ksz_half;
	}

	// result = new arma::Mat<T>(rows_out, cols_out);


}

