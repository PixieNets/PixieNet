//
// Created by Zal on 11/21/16.
//

#pragma once

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <armadillo>
#include "BinaryConvolution.h"

#define StrDblPair std::pair<std::string, double>
#define VecStrDblPair std::vector<StrDblPair>

using namespace bconv;

class XnorNetwork {
    std::vector<std::string>            XN_labels;
    int                                 XN_totalLabels;
    std::vector<BinaryConvolution>      XN_convLayers;

public:
    XnorNetwork();
    ~XnorNetwork();

    void            buildAlexNet();
    void            buildMiniNet();
    arma::vec       forwardPass(arma::cube image);


    void            loadLabelsFromFile(std::string path);
    VecStrDblPair   getTopNLabels(int N, arma::vec outputVec);

    arma::vec       softmax(arma::mat z);
};

