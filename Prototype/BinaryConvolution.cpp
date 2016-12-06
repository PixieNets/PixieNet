//
// Created by Esha Uboweja on 11/22/16.
//

#include "BinaryConvolution.h"

#include <assert.h>

using namespace bd;
using namespace bconv;

BinaryConvolution::BinaryConvolution(uint w, uint h, uint ch, uint k, uint stride, Convolution conv_type,
                                     Nonlinearity actv_type, Pooling pool_type, uint pool_size, uint pool_stride) {

    this->init_convolution(w, h, ch, k, stride, conv_type);
    this->init_pooling(pool_type, pool_size, pool_stride);
    this->init_nonlinearity(actv_type);
}

void BinaryConvolution::init_convolution(uint w, uint h, uint ch, uint k, uint stride, Convolution conv_type) {

    if (w <= 0 || h <= 0 || ch <= 0 || k <= 0 || stride <= 0) {
        throw std::invalid_argument("[BinaryConvolution::init_convolution] Convolution weights matrix dimensions, stride should be positive");
    }
    this->bc_width = w;
    this->bc_height = h;
    this->bc_channels = ch;
    this->bc_filters = k;
    this->bc_conv_stride = stride;

    if (conv_type == Convolution::same) {
        this->bc_padding = w / 2;
    }
    if (conv_type == Convolution::valid) {
        this->bc_padding = 0;
    }
    // The convolution filter k for computing the scales of each input sub-tensor
    this->bc_box_filter = arma::ones<arma::mat>(w, h) * (1.0 / (w * h));

    // The 4D hyper-cube weights of the convolution layer
    this->bc_conv_weights.reserve(this->bc_filters);
    for (uint f = 0; f < this->bc_filters; ++f) {
        this->bc_conv_weights.emplace_back(BinaryTensor3D(this->bc_height, this->bc_width, this->bc_channels, BIT_ZERO));
    }
}

void BinaryConvolution::init_pooling(Pooling pool_type, uint pool_size, uint pool_stride) {
    if (pool_size <= 0 || pool_stride <= 0) {
        throw std::invalid_argument("[BinaryConvolution::init_pooling] Pooling kernel dimensions, stride should be positive");
    }
    if (pool_type == Pooling::none) {
        this->bc_pool = false;
    }
    this->bc_pool_type = pool_type;
    this->bc_pool_size = pool_size;
    this->bc_pool_stride = pool_stride;
}

void BinaryConvolution::init_nonlinearity(Nonlinearity actv_type) {
    if (actv_type == Nonlinearity::none) {
        this->bc_nonlinear_actv = false;
    }
    this->bc_nonlinearity = actv_type;
}

BinaryConvolution::~BinaryConvolution() {
    // Explicit destructor only for pointer members
}

arma::mat BinaryConvolution::normalizeData2D(arma::mat data) {
    arma::mat norm_data = (data - arma::mean(arma::mean(data))) / arma::stddev(arma::stddev(data));
    return norm_data;
}

arma::cube BinaryConvolution::normalizeData3D(arma::cube data) {
    arma::cube norm_data = arma::zeros<arma::cube>(arma::size(data));
    for (uint ch = 0; ch < data.n_slices; ++ch) {
        norm_data.slice(ch) = normalizeData2D(data.slice(ch));
    }
    return norm_data;
}

arma::mat BinaryConvolution::input2KMat(arma::cube norm_data) {
    // A = (\sum_{i=1}^n I(:, :, i)) / n
    arma::mat A = arma::mean(norm_data, 2);
    // Convolve with box filter k of size w x h of the convolution weights
    // Discuss if Box filter using integral images is a good idea (will it be faster here
    // because the input keeps changing?)
    arma::mat K = arma::conv2(A, this->bc_box_filter, "same");
    return K;
}

BinaryTensor3D BinaryConvolution::binarizeInput(arma::cube norm_data) {
    return BinaryTensor3D(norm_data);
}

arma::cube BinaryConvolution::doBinaryConv(BinaryTensor3D input, arma::mat K) {

    // (sign(I) xnor_conv sign(W)) xnor_prod K,w_alpha
    arma::cube output;

    if (input.cols() < this->bc_width || input.rows() < this->bc_height) {
        // result is an empty matrix
        return output;
    }

    if (input.channels() != this->bc_channels) {
        std::cerr << "[BinaryConv::doBinConv] Input (arg1) and conv weights should have the same number of channels\n";
        return output;
    }

    // Output dimensions
    uint n_filter = this->bc_height * this->bc_width;
    uint rows_out = (input.rows() - this->bc_height + 2 * this->bc_padding) / this->bc_conv_stride + 1;
    uint cols_out = (input.cols() - this->bc_width + 2 * this->bc_padding) / this->bc_conv_stride + 1;
    output = arma::zeros(rows_out, cols_out, this->bc_filters);
    output.zeros();


    // Simple for-loop implementation
    std::vector<BinaryLayer*> inputVec = input.tensor();
    for (uint f = 0; f < this->bc_filters; ++f) {
        BinaryTensor3D cur_weights = this->bc_conv_weights[f];
        for (uint ch = 0; ch < this->bc_channels; ++ch) {
            // 1 (a). Spatial column layout of input
            BinaryLayer col_input = inputVec[ch]->im2col(this->bc_width, this->bc_height,
                                                     this->bc_padding, this->bc_conv_stride);
            // 1. XNOR Product of input and weights;
            // 1 (b). Spatial row layout of weight filter
            BinaryLayer wt_input = cur_weights.tensor()[ch]->reshape(1, n_filter).repmat(col_input.height(), 1);
            // 1 (c). XNOR product
            BinaryLayer result = col_input * wt_input;
            // 2. Bitcount and reshape
            output.slice(f) += result.binMtx()->bitCountPerRow(true, rows_out, cols_out);
        }
        // Element-wise multiply by scalar factors of input tensor and weights alpha
        output.slice(f) = (output.slice(f) % K) * cur_weights.alpha();
    }


    return output;
}

arma::cube BinaryConvolution::nonLinearActivate(arma::cube data) {
    arma::cube output = data;
    if (this->bc_nonlinearity == Nonlinearity::relu) {
        output.elem(arma::find(data < 0)).zeros();
    }
    return output;
}

arma::mat BinaryConvolution::poolMat(arma::mat data) {
    uint width = (uint) (data.n_cols - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint height = (uint) (data.n_rows - this->bc_pool_size) / this->bc_pool_stride - 1;

    arma::mat output = arma::zeros<arma::mat>(height, width);
    for (uint row = 0; row < height; row += this->bc_pool_stride) {
        for (uint col = 0; col < width; col += this->bc_pool_stride) {
            uint row_start = row * this->bc_pool_size;
            uint row_end = row_start + this->bc_pool_size - 1;
            uint col_start = col * this->bc_pool_size;
            uint col_end = col_start + this->bc_pool_size - 1;
            if (this->bc_pool_type == Pooling::max) {
                output(row, col) = arma::max(arma::max(data(arma::span(row_start, row_end),
                                                            arma::span(col_start, col_end))));
            } else if (this->bc_pool_type == Pooling::min) {
                output(row, col) = arma::min(arma::min(data(arma::span(row_start, row_end),
                                                            arma::span(col_start, col_end))));
            } else if (this->bc_pool_type == Pooling::average) {
                output(row, col) = arma::mean(arma::mean(data(arma::span(row_start, row_end),
                                                              arma::span(col_start, col_end))));
            }
        }
    }
    return output;
}

arma::cube BinaryConvolution::doPooling(arma::cube data) {
    uint width = (uint) (data.n_cols - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint height = (uint) (data.n_rows - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint channels = (uint) data.n_slices;

    arma::cube output = arma::zeros<arma::cube>(height, width, channels);
    for (uint ch = 0; ch < channels; ++ch) {
        output.slice(ch) = poolMat(data.slice(ch));
    }
    return output;
}

void BinaryConvolution::setWeights(BinaryTensor4D conv_weights) {
    if (conv_weights.empty()) {
        throw std::invalid_argument("[BinaryConvolution::setWeights] Input set of conv_weights must be non-empty");
    }
    this->bc_width = conv_weights[0].cols();
    this->bc_height = conv_weights[0].rows();
    this->bc_channels = conv_weights[0].channels();
    this->bc_filters = (uint) conv_weights.size();
    this->bc_conv_weights.reserve(this->bc_channels);
    for (uint f = 0; f < this->bc_filters; ++f) {
        this->bc_conv_weights.emplace_back(BinaryTensor3D(conv_weights[f]));
    }
}

arma::cube BinaryConvolution::forwardPass(arma::cube data) {

    if (data.empty()) {
        throw std::invalid_argument("[BinaryConvolution::forwardPass] Input data must be non-empty!");
    }
    if (data.n_slices != this->bc_channels) {
        std::string err = std::string("BinaryConvolution::forwardPass] Input data #channels = ")
                          + std::to_string(data.n_slices)
                          + std::string(" must match convolution weights channels = ")
                          + std::to_string(this->bc_channels);
    }

    // 1. Normalize input
    arma::cube norm_data = this->normalizeData3D(data);
    // 2. Generate K matrix containing scalar factors for each input sub-tensor
    arma::mat K = this->input2KMat(norm_data);
    // 3. Binarize normalized data - activation
    BinaryTensor3D input = this->normalizeData3D(norm_data);
    // 4. Perform the binary convolution
    arma::cube result = this->doBinaryConv(input, K);
    // 5. Apply non-linearity
    if (this->bc_nonlinear_actv) {
        result = this->nonLinearActivate(result);
    }
    // 6. Apply pooling
    if (this->bc_pool) {
        result = this->doPooling(result);
    }

    return result;
}

BinaryTensor4D BinaryConvolution::randomTensor4D(uint width, uint height, uint channels, uint filters, uint nrandom) {
    BinaryTensor4D result;
    result.reserve(filters);
    for (uint f = 0; f < filters; ++f) {
        result.emplace_back(BinaryTensor3D(height, width, channels, 1.0, true, nrandom));
    }
    return result;
}

std::string BinaryConvolution::bt4ToString(BinaryTensor4D input) {
    std::string result = "";
    for (uint f = 0; f < input.size(); ++f) {
        result += "[FILTER #" + std::to_string(f) + "]\n";
        result += input[f].toString() + "\n";
    }
    return result;
}
