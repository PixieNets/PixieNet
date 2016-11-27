//
// Created by Zal on 11/19/16.
//

#pragma once

#include "BinaryMatrix.h"

#include <armadillo>

typedef unsigned int uint;

namespace bd {
    class BinaryLayer;
    // Concatenate multiple binary layer channels to form a 3D binary tensor
    typedef BinaryLayer** BinaryTensor;
}

class bd::BinaryLayer {
private:
    uint            bl_width;
    uint            bl_height;
    BinaryMatrix    *bl_binMtx;
    double          bl_alpha;

public:
    BinaryLayer(uint w, uint h);
    ~BinaryLayer();

    void binarizeMat(arma::mat data);
    void binarizeWeights(double *weights, int size);
    void getDoubleWeights(double **weights, int *size);

    // Accessor functions for data members
    uint width() {   return bl_width; }
    uint height() {   return bl_height; }
    BinaryMatrix* binMtx() {   return bl_binMtx; }
    double alpha() {    return bl_alpha; }

};
