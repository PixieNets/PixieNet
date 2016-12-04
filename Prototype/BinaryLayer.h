//
// Created by Zal on 11/19/16.
//

#pragma once

#include "BinaryMatrix.h"

#include <armadillo>
#include <vector>


namespace bd {
    class BinaryLayer;
    // Concatenate multiple binary layer channels to form a 3D binary tensor
    typedef std::vector<BinaryLayer*>     BinaryTensor3D;
    // Concatenate multiple 3D binary tensors to form a 4D binary tensor
    typedef std::vector<BinaryTensor3D>   BinaryTensor4D;
}

class bd::BinaryLayer {
private:
    uint            bl_width;
    uint            bl_height;
    BinaryMatrix    *bl_binMtx;
    double          bl_alpha;

public:
    BinaryLayer(uint w, uint h);
    BinaryLayer(arma::umat input2D);
    ~BinaryLayer();

    void        binarizeMat(arma::mat data);
    void        binarizeWeights(double *weights, int size);
    void        getDoubleWeights(double **weights, int *size);
    BinaryLayer operator*(const BinaryLayer &other);
    BinaryLayer im2col(uint block_width, uint block_height, uint padding, uint stride);
    BinaryLayer repmat(uint n_rows, uint n_cols);
    BinaryLayer reshape(uint new_rows, uint new_cols);


    // Accessor functions for data members
    uint width()            {   return bl_width;  }
    uint height()           {   return bl_height; }
    BinaryMatrix* binMtx()  {   return bl_binMtx; }
    double alpha()          {   return bl_alpha;  }

};
