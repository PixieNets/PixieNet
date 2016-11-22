//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include <cmath>
#include "BinaryLayer.h"

BinaryLayer::BinaryLayer(int w, int h) {
    this->binMtx = new BinaryMatrix(w, h);
    this->alpha = 1.0;
}

BinaryLayer::~BinaryLayer() {
    if(this->binMtx != nullptr)
        delete this->binMtx;
}

/**
 * Converts double precision weights matrix to single-bit representation
 * binary weights matrix
 * @param weights - double precision weights values matrix
 * @param size - #elements in the weights matrix
 */
void BinaryLayer::binarizeWeights(double *weights, int size) {
    assert(size == (this->binMtx->width * this->binMtx->height));

    double alpha = 0.0;
    for (int i = 0;  i < size; ++i) {
        alpha += std::fabs(weights[i]);
        this->binMtx->setValueAt(i, (weights[i] >= 0)? BIT_ONE:BIT_ZERO);
    }
    this->alpha = alpha / size;
}

/**
 *
 * @param weights
 * @param size
 */
void BinaryLayer::getDoubleWeights(double **weights, int *size) {
    if(*weights == nullptr) {
        *weights = new double[binMtx->width*binMtx->height];
        *size = binMtx->width*binMtx->height;
    }
    else {
        assert(binMtx->width*binMtx->height == *size);
    }
    for(int i=0; i<*size; ++i) {
        *weights[i] = this->binMtx->getValueAt(i);
    }
}