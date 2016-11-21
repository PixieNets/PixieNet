//
// Created by Zal on 11/19/16.
//

#include <assert.h>
#include "BinaryLayer.h"

BinaryLayer::BinaryLayer(int w, int h) {
    this->binMtx = new BinaryMatrix(w, h);
    this->scale = 1.0;
}

BinaryLayer::~BinaryLayer() {
    if(binMtx != nullptr)  delete binMtx;
}

void BinaryLayer::binarizeWeights(double* weights, int size) {
    assert(size == this->binMtx->width*this->binMtx->height);

    double acc = 0.0;
    for(int i=0; i<size; ++i) {
        acc += weights[i];
        this->binMtx->setValueAt(i, (weights[i]>=0.0)? 1:0);
    }
    this->scale /=size;
}

void BinaryLayer::getDoubleWeights(double** weights, int* size) {
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