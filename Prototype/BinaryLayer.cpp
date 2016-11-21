//
// Created by Zal on 11/19/16.
//

#include "BinaryLayer.h"

BinaryLayer::BinaryLayer() {

}

BinaryLayer::~BinaryLayer() {
    if(binMtx != NULL)  delete binMtx;
}

void BinaryLayer::binarizeWeights(double* weights) {

}

double* BinaryLayer::getDoubleWeights() {
    return nullptr;
}