//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <cmath>

#include "BinaryLayer.h"

using namespace bd;

BinaryLayer::BinaryLayer(uint w, uint h) {
    this->init(w, h, 1.0);
    this->bl_binMtx = new BinaryMatrix(w, h);
}

BinaryLayer::BinaryLayer(uint w, uint h, uint8 value) {
    this->init(w, h, 1.0);
    this->bl_binMtx = new BinaryMatrix(w, h, value);
}

BinaryLayer::BinaryLayer(arma::mat input2D) {
    this->init((uint) input2D.n_cols, (uint) input2D.n_rows);
    this->binarizeMat(input2D);
}

BinaryLayer::BinaryLayer(arma::umat input2D) {
    this->init((uint) input2D.n_cols, (uint) input2D.n_rows);
    this->bl_binMtx = new BinaryMatrix(input2D);
    this->bl_alpha = arma::mean(arma::mean(arma::abs(input2D)));
}

BinaryLayer::BinaryLayer(BinaryMatrix bm, double alpha) {
    this->init(bm.width(), bm.height(), alpha);
    this->bl_binMtx = new BinaryMatrix(bm);
}

BinaryLayer::BinaryLayer(uint w, uint h, double alpha, bool randomized, uint n) {
    this->init(w, h, alpha);
    this->bl_binMtx = new BinaryMatrix(w, h, randomized, n);
}

BinaryLayer::BinaryLayer(const BinaryLayer &bl) {
    this->copy(bl);
}

void BinaryLayer::copy(const BinaryLayer &other) {
    this->init(other.bl_width, other.bl_height, other.bl_alpha);
    // Deep copy the matrix
    if(this->bl_binMtx != nullptr){
        delete this->bl_binMtx;
    }
    this->bl_binMtx = new BinaryMatrix(*(other.bl_binMtx));
}

void BinaryLayer::init(uint width, uint height, double alpha) {
    this->bl_width = width;
    this->bl_height = height;
    this->bl_alpha = alpha;
}

BinaryLayer::~BinaryLayer() {
    if(this->bl_binMtx != nullptr)
        delete this->bl_binMtx;
}

/**
 * Converts double precision 2D weights matrix in Armadillo to single
 * bit representation binary weights matrix
 * @param data - double precision 2D weights matrix
 */
void BinaryLayer::binarizeMat(arma::mat data) {
    if (this->bl_width != data.n_cols || this->bl_height != data.n_rows) {
        std::string err = std::string("[BinaryLayer::binarizeMat] Input arma mat (") + std::to_string(data.n_rows)
                          + ", " + std::to_string(data.n_cols) + std::string(") should have same size as binary layer (")
                          + std::to_string(this->bl_height) + std::string(", ") + std::to_string(this->bl_width)
                          + std::string("). Invalid input!");
        throw std::invalid_argument(err);
    }

    this->bl_binMtx = new BinaryMatrix(this->bl_width, this->bl_height);
    uint n_elems = this->bl_width * this->bl_height;
    for (uint row = 0; row < data.n_rows; ++row) {
        for (uint col = 0; col < data.n_cols; ++col) {
            uint8 result = (data(row, col) >= 0.0) ? BIT_ONE : BIT_ZERO;
            this->bl_binMtx->setValueAt(row, col, result);
        }
    }
    this->bl_alpha = arma::accu(arma::abs(data))/ n_elems;
}

/**
 * Converts double precision weights matrix to single-bit representation
 * binary weights matrix
 * @param weights - double precision weights values matrix
 * @param size - #elements in the weights matrix
 */
void BinaryLayer::binarizeWeights(double *weights, int size) {
    assert(size == (this->bl_binMtx->width() * this->bl_binMtx->height()));

    double bl_alpha = 0.0;
    for (uint i = 0;  i < size; ++i) {
        bl_alpha += std::fabs(weights[i]);
        this->bl_binMtx->setValueAt(i, (weights[i] >= 0)? BIT_ONE:BIT_ZERO);
    }
    this->bl_alpha = bl_alpha / size;
}

/**
 *
 * @param weights
 * @param size
 */
void BinaryLayer::getDoubleWeights(double **weights, int *size) {
    if(*weights == nullptr) {
        *weights = new double[bl_binMtx->width()*bl_binMtx->height()];
        *size = bl_binMtx->width()*bl_binMtx->height();
    }
    else {
        assert(bl_binMtx->width()*bl_binMtx->height() == *size);
    }
    for(int i = 0; i < *size; ++i) {
        *weights[i] = this->bl_binMtx->getValueAt(i);
    }
}

BinaryLayer& BinaryLayer::operator=(const BinaryLayer &rhs) {
    // Only do assignment if RHS is a different object from this.
    if (this != &rhs) {
        this->copy(rhs);
    }
    return *this;
}

BinaryLayer BinaryLayer::operator*(const BinaryLayer&other) {
    BinaryMatrix resultMtx = (*(this->bl_binMtx)) * (*(other.bl_binMtx));
    BinaryLayer result = BinaryLayer(resultMtx, this->bl_alpha * other.bl_alpha);
    return result;
}

BinaryLayer BinaryLayer::im2col(uint block_width, uint block_height, uint padding, uint stride) {
    BinaryMatrix resultMtx = this->bl_binMtx->im2col(block_width, block_height, padding, stride);
    BinaryLayer result = BinaryLayer(resultMtx, this->bl_alpha);
    return result;
}

BinaryLayer BinaryLayer::repmat(uint n_rows, uint n_cols) {
    // Note that after repmat, the alpha value is not correct
    BinaryMatrix resultMtx = this->bl_binMtx->repmat(n_rows, n_cols);
    BinaryLayer result = BinaryLayer(resultMtx);
    return result;
}

BinaryLayer BinaryLayer::reshape(uint new_rows, uint new_cols) {
    BinaryMatrix resultMtx = this->bl_binMtx->reshape(new_rows, new_cols);
    BinaryLayer result = BinaryLayer(resultMtx, this->bl_alpha);
    return result;
}
