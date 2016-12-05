//
// Created by Esha Uboweja on 12/4/16.
//

#include "BinaryTensor3D.h"

using namespace bd;

BinaryTensor3D::BinaryTensor3D(uint rows, uint cols, uint channels, uint8 value, double alpha) {
    this->init(rows, cols, channels, alpha);
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        this->bt3_tensor.emplace_back(new BinaryLayer(this->bt3_cols, this->bt3_rows, value));
    }
}

BinaryTensor3D::BinaryTensor3D(uint rows, uint cols, uint channels, double alpha, bool randomized, uint n) {
    this->init(rows, cols, channels, alpha);
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        this->bt3_tensor.emplace_back(new BinaryLayer(this->bt3_cols, this->bt3_rows, 1.0, randomized, n));
    }
}

BinaryTensor3D::BinaryTensor3D(arma::ucube tensor) {
    this->init((uint) tensor.n_rows, (uint) tensor.n_cols, (uint) tensor.n_slices);
    uint n_elems = this->bt3_rows * this->bt3_cols * this->bt3_channels;

    this->bt3_tensor.reserve(this->bt3_channels);
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        this->bt3_tensor.emplace_back(new BinaryLayer(tensor.slice(ch)));
    }
    this->bt3_alpha = arma::accu(arma::abs(tensor)) / (double) n_elems;
}

BinaryTensor3D::BinaryTensor3D(const BinaryTensor3D &tensor) {
    this->init(tensor.bt3_rows, tensor.bt3_cols, tensor.bt3_channels, tensor.bt3_alpha);
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        this->bt3_tensor.emplace_back(new BinaryLayer(*(tensor.bt3_tensor[ch])));
    }
}

BinaryTensor3D::~BinaryTensor3D() {
    // Delete each layer
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        if (this->bt3_tensor[ch]) { // Not NULL
            delete this->bt3_tensor[ch];
        }
    }
}

void BinaryTensor3D::init(uint rows, uint cols, uint channels, double alpha) {
    this->bt3_rows = rows;
    this->bt3_cols = cols;
    this->bt3_channels = channels;
    this->bt3_alpha = alpha;
    this->bt3_tensor.reserve(this->bt3_channels);
}

BinaryLayer BinaryTensor3D::im2col(uint block_width, uint block_height, uint padding, uint stride) {
    uint block_ht_half = block_height / 2;
    uint block_wd_half = block_width / 2;
    if (padding > block_ht_half || padding > block_wd_half) {
        throw std::invalid_argument("[BinaryTensor3D::im2col] padding (arg3), block_width(arg1) and block_height (arg2) are invalid");
    }
    uint n = block_width * block_height;

    uint rows_out = (uint) (this->bt3_rows - block_height + 2 * padding);
    if (rows_out % stride) {
        throw std::invalid_argument("[BinaryTensor3D::im2col] block_height (arg2), padding (arg3) and stride (arg4) are invalid");
    }
    rows_out = rows_out / stride + 1;
    uint cols_out = (uint) (this->bt3_cols - block_width + 2 * padding);
    if (cols_out % stride) {
        throw std::invalid_argument("[BinaryTensor3D::im2col] block_width (arg1), padding (arg3) and stride (arg4) are invalid");
    }
    cols_out = cols_out / stride + 1;

    uint result_height = rows_out * cols_out;
    uint result_width = n * this->bt3_channels;

    // Implementation 1
    BinaryLayer result(result_width, result_height, this->bt3_alpha);
    uint start_idx = 0, end_idx = result_width;
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        BinaryMatrix bm = this->bt3_tensor[ch]->binMtx()->im2col(block_width, block_height, padding, stride);
        // Copy the im2col result for this channel
        start_idx = ch * n;
        end_idx = start_idx + n;
        for (uint row = 0; row < result_height; ++row) {
            for (uint col = start_idx; col < end_idx; ++col) {
                result.binMtx()->setValueAt(row, col, bm.getValueAt(row - start_idx, col));
            }
        }
    }

    return result;
}
