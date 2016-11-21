//
// Created by Zal on 11/19/16.
//

#ifndef PROTOTYPE_BINARYLAYER_H
#define PROTOTYPE_BINARYLAYER_H

#include "BinaryMatrix.h"

class BinaryLayer {
    BinaryMatrix    *binMtx;
    double          scale;

public:
    BinaryLayer(int w, int h);
    ~BinaryLayer();

    void binarizeWeights(double* weights, int size);
    void getDoubleWeights(double** weights, int* size);

};


#endif //PROTOTYPE_BINARYLAYER_H
