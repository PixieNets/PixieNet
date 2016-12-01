//
// Created by Esha Uboweja on 11/22/16.
//

#include "BinaryConvolution.h"

#include <assert.h>

BinaryConvolution::BinaryConvolution(uint w, uint h, uint ch, uint k, uint stride,
                                     uint padding, bool pool, Pooling pool_type,
                                     uint pool_size, uint pool_stride) {
    assert(w > 0 && h > 0 && ch > 0 && stride > 0);

    this->bc_width = w;
    this->bc_height = h;
    this->bc_channels = ch;
    this->bc_filters = k;
    this->bc_stride = stride;
    this->bc_padding = padding;
    this->bc_pool = pool;
    this->bc_pool_type = pool_type;
    this->bc_pool_size = pool_size;
    this->bc_pool_stride = pool_stride;

    // The convolution filter k for computing the scales of each input sub-tensor
    this->bc_box_filter = arma::ones<arma::mat>(w, h) * (1.0 / (w * h));

    // The 4D hyper-cube weights of the convolution layer
    this->bc_conv_weights = new BinaryTensor3D[this->bc_filters];
    for (uint i = 0; i < this->bc_filters; ++i) {
        this->bc_conv_weights[i] = new BinaryLayer*[this->bc_channels];
        for (uint j = 0; j < this->bc_channels; ++j) {
            this->bc_conv_weights[i][j] = new BinaryLayer(w, h);
        }
    }
}

BinaryConvolution::~BinaryConvolution() {
    // Weights matrix
    for (uint i = 0; i < this->bc_filters; ++i) {
        // delete each member of the array
        for (uint j = 0; j < this->bc_channels; ++j) {
            delete this->bc_conv_weights[i][j];
        }
    }
    // delete the array
    delete[] bc_conv_weights;
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
    uint width = norm_data.n_cols;
    uint height = norm_data.n_rows;
    uint channels = norm_data.n_slices;
    BinaryTensor3D tensor = new BinaryLayer*[channels];
    for (int ch = 0; ch < channels; ++ch) {
        tensor[ch] = new BinaryLayer(width, height);
        tensor[ch]->binarizeMat(norm_data.slice(ch));
    }
    return tensor;
}

arma::cube BinaryConvolution::doBinaryConv(BinaryTensor4D input, arma::mat K) {

    // (sign(I) xnor_conv sign(W)) xnor_prod K,w_alpha
    arma::cube output;



    return output;
}

arma::mat BinaryConvolution::poolMat(arma::mat data) {
    uint width = (data.n_cols - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint height = (data.n_rows - this->bc_pool_size) / this->bc_pool_stride - 1;

    arma::mat output = arma::zeros<arma::mat>(height, width);
    for (int row = 0; row < height; row += this->bc_pool_stride) {
        for (int col = 0; col < width; col += this->bc_pool_stride) {
            int row_start = row * this->bc_pool_size;
            int row_end = row_start + this->bc_pool_size - 1;
            int col_start = col * this->bc_pool_size;
            int col_end = col_start + this->bc_pool_size - 1;
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
    uint width = (data.n_cols - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint height = (data.n_rows - this->bc_pool_size) / this->bc_pool_stride - 1;
    uint channels = data.n_slices;

    arma::cube output = arma::zeros<arma::cube>(height, width, channels);
    for (uint ch = 0; ch < channels; ++ch) {
        output.slice(ch) = poolMat(data.slice(ch));
    }
    return output;
}
