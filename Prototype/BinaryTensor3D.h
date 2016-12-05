//
// Created by Esha Uboweja on 12/4/16.
//

#pragma once

#include "BinaryLayer.h"
#include <vector>
#include <armadillo>

class bd::BinaryTensor3D {
private:
    uint                      bt3_rows;
    uint                      bt3_cols;
    uint                      bt3_channels;
    double                    bt3_alpha;
    std::vector<BinaryLayer*> bt3_tensor;

public:
    // Initialize using a fixed value
    BinaryTensor3D(uint rows, uint cols, uint channels, uint8 value);
    // Initialize to random values
    BinaryTensor3D(uint rows, uint cols, uint channels, bool randomized=false, uint n=0);
    // Initialize using Armadillo cube
    BinaryTensor3D(arma::cube tensor);
    // Initialize using copy constructor
    BinaryTensor3D(const BinaryTensor3D &tensor);
    // Destructor
    ~BinaryTensor3D();

    // im2col(3D tensor) returns a 2D matrix
    BinaryLayer im2col(uint block_width, uint block_height, uint padding, uint stride);

    // accessor functions for members
    uint                      rows()     {    return bt3_rows;        }
    uint                      cols()     {    return bt3_cols;        }
    uint                      channels() {    return bt3_channels;    }
    double                    alpha()    {    return bt3_alpha;       }
    std::vector<BinaryLayer*> tensor()   {    return bt3_tensor;      }
};


#endif //PROTOTYPE_BINARYTENSOR3D_H
