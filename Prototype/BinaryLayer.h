//
// Created by Zal on 11/19/16.
//

#pragma once

#include <armadillo>

#include "BinaryMatrix.h"

class BinaryLayer {
private:
    int             bl_width;
    int             bl_height;
    BinaryMatrix    *bl_binMtx;
    double          bl_alpha;

public:
    BinaryLayer(int w, int h);
    ~BinaryLayer();

    void binarizeMat(arma::mat data);
    void binarizeWeights(double *weights, int size);
    void getDoubleWeights(double **weights, int *size);

    // Accessor functions for data members
    int width() {   return bl_width; }
    int height() {   return bl_height; }
    BinaryMatrix* binMtx() {   return bl_binMtx; }
    double alpha() {    return bl_alpha; }

};
