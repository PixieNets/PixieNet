//
// Created by Zal on 11/21/16.
//

#pragma once

#include <armadillo>

#include "BinaryLayer.h"

using namespace bd;

class XnorNetUtils {
public:
    BinaryMatrix* centerDataMat(arma::mat data);
    BinaryTensor normalizeData(arma::cube data);
};
