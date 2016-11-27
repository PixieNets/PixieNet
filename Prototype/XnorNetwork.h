//
// Created by Zal on 11/21/16.
//

#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <vector>


class XnorNetwork {
    std::vector<std::string> XN_labels;

    void loadLabelsFromFile(std::string path);
};

