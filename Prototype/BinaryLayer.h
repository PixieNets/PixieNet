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
    BinaryLayer();
    ~BinaryLayer();

    void binarizeWeights(double* weights);
    double* getDoubleWeights();

};


#endif //PROTOTYPE_BINARYLAYER_H
