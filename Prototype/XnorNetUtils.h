//
// Created by Zal on 11/21/16.
//

#pragma once

#include <armadillo>

#include "BinaryLayer.h"
#include "BinaryTensor3D.h"

using namespace bd;

class XnorNetUtils {
public:
    BinaryMatrix*  centerDataMat(arma::mat data);
    BinaryTensor3D normalizeData3D(arma::cube data);

    arma::vec softmax(arma::mat weight, arma::vec prevOutput);
};
