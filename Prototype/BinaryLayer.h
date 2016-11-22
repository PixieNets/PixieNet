//
// Created by Zal on 11/19/16.
//

#pragma once

#include "BinaryMatrix.h"

class BinaryLayer {
    BinaryMatrix    *binMtx;
    double          alpha;

public:
    BinaryLayer(int w, int h);
    ~BinaryLayer();

    void binarizeWeights(double *weights, int size);
    void getDoubleWeights(double **weights, int *size);

};
