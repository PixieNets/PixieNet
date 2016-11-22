//
// Created by Zal on 11/21/16.
//

#pragma once

#include <armadillo>

#include "BinaryMatrix.h"

#define BMatArr BinaryMatrix**

class XnorNetUtils {
public:
    BinaryMatrix* centerDataMat(arma::mat data);
    BMatArr normalizeData(arma::cube data);
};
