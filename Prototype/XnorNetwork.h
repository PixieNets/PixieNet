//
// Created by Zal on 11/21/16.
//

#pragma once
#include <armadillo>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>

#define StrDblPair std::pair<std::string, double>
#define VecStrDblPair std::vector<StrDblPair>

class XnorNetwork {
    std::vector<std::string>    XN_labels;
    int                         XN_totalLabels;

    XnorNetwork();
    ~XnorNetwork();

    void            loadLabelsFromFile(std::string path);
    VecStrDblPair   getTopNLabels(int N, arma::vec outputVec);
};

