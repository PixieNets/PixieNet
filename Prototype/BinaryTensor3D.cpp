//
// Created by Esha Uboweja on 12/4/16.
//

#include "BinaryTensor3D.h"

using namespace bd;

BinaryTensor3D::BinaryTensor3D(uint rows, uint cols, uint channels, uint8 value, double alpha) {
    this->bt3_rows = rows;
    this->bt3_cols = cols;
    this->bt3_channels = channels;
    this->bt3_alpha = alpha;
    this->bt3_tensor.reserve(this->bt3_channels);
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        this->bt3_tensor.emplace_back(new BinaryLayer(this->bt3_cols, this->bt3_rows, value));
    }
}

BinaryTensor3D::BinaryTensor3D(uint rows, uint cols, uint channels, double alpha, bool randomized, uint n) {
    this->bt3_rows = rows;
    this->bt3_cols = cols;
    this->bt3_channels = channels;
    this->bt3_alpha = alpha;
    this->bt3_tensor.reserve(this->bt3_channels);
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        this->bt3_tensor.emplace_back(new BinaryLayer(this->bt3_cols, this->bt3_rows, 1.0, randomized, n));
    }
}

BinaryTensor3D::BinaryTensor3D(arma::ucube tensor) {
    this->bt3_rows = (uint) tensor.n_rows;
    this->bt3_cols = (uint) tensor.n_cols;
    this->bt3_channels = (uint) tensor.n_slices;
    uint n_elems = this->bt3_rows * this->bt3_cols * this->bt3_channels;

    this->bt3_tensor.reserve(this->bt3_channels);
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        this->bt3_tensor.emplace_back(new BinaryLayer(tensor.slice(ch)));
    }
    this->bt3_alpha = arma::accu(arma::abs(tensor)) / (double) n_elems;
}

BinaryTensor3D::BinaryTensor3D(const BinaryTensor3D &tensor) {

}

BinaryTensor3D::~BinaryTensor3D() {
    // Delete each layer
    for (uint ch = 0; ch < this->bt3_channels; ++ch) {
        if (this->bt3_tensor[ch]) { // Not NULL
            delete this->bt3_tensor[ch];
        }
    }
}

BinaryLayer BinaryTensor3D::im2col(uint block_width, uint block_height, uint padding, uint stride) {
    BinaryLayer result(0,0);
    return result;
}
