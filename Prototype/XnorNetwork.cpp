//
// Created by Zal on 11/21/16.
//

#include "XnorNetwork.h"
#include "assert.h"

XnorNetwork::XnorNetwork() {

}

XnorNetwork::~XnorNetwork() {

}

/*
 * This is for prototype testis purposes and needs to be changed
 * to Objective-C for loading in the iPhone
 */
void XnorNetwork::loadLabelsFromFile(std::string path) {
    std::ifstream labelsFile(path);

    if(!labelsFile.is_open()) {
        std::cout << "Could not open file " << path << std::endl;
        return;
    }

    std::string tmpLabel;
    for(int i=0; !labelsFile.eof(); ++i) {
        labelsFile >> tmpLabel;
        this->XN_labels.push_back(tmpLabel);
    }
    this->XN_totalLabels = this->XN_labels.size();
    std::cout << "Loaded " << this->XN_totalLabels << "labels" << std::endl;
}

VecStrDblPair XnorNetwork::getTopNLabels(int N, arma::vec outputVec) {
    VecStrDblPair rankedVec;

    //Obtain the indices in descending order
    arma::uvec rankedIdx = arma::sort_index(outputVec, "descend");
    int entryIdx;
    for(int rank=0; rank<N; ++rank) {
        entryIdx = rankedIdx(rank);
        StrDblPair newEntry = StrDblPair(this->XN_labels[entryIdx], outputVec(entryIdx));
        rankedVec.push_back(newEntry);
    }
    return rankedVec;
}

void XnorNetwork::buildAlexNet() {

    XN_convLayers.push_back(BinaryConvolution(11, 11, 3, 96, 4, Convolution::same, Nonlinearity::relu, Pooling::max, 3, 2));
    XN_convLayers.push_back(BinaryConvolution(5, 5, 96, 256, 3, Convolution::same, Nonlinearity::relu, Pooling::max, 3, 2));
    XN_convLayers.push_back(BinaryConvolution(3, 3, 256, 384, 1, Convolution::same, Nonlinearity::relu, Pooling::none));
    XN_convLayers.push_back(BinaryConvolution(3, 3, 384, 384, 1, Convolution::same, Nonlinearity::relu, Pooling::none));
    XN_convLayers.push_back(BinaryConvolution(3, 3, 384, 256, 1, Convolution::same, Nonlinearity::relu, Pooling::max, 3, 2));
    XN_convLayers.push_back(BinaryConvolution(6, 6, 256, 4096, 1, Convolution::same, Nonlinearity::relu, Pooling::none));
    XN_convLayers.push_back(BinaryConvolution(1, 1, 4096, 1000, 1, Convolution::same, Nonlinearity::relu, Pooling::none));
}

void XnorNetwork::buildMiniNet() {
//    XN_convLayers.push_back(BinaryConvolution(11, 11, 3, 3, 1, Convolution::same, Nonlinearity::relu, Pooling::max, 2, 2));
//    XN_convLayers.push_back(BinaryConvolution(11, 11, 3, 96, 1, Convolution::same, Nonlinearity::relu, Pooling::max, 2, 2));
    XN_convLayers.push_back(BinaryConvolution(11, 11, 3, 96, 4, Convolution::valid, Nonlinearity::relu, Pooling::max, 2, 2));
}

arma::vec XnorNetwork::forwardPass(arma::cube image){
    arma::vec classProbs;
    int numLayers = XN_convLayers.size();
    if (numLayers < 1){
        return zeros<vec>(0);
    }
    arma::cube lastRes = image;

    for(int fwdPos = 0; fwdPos < numLayers; ++fwdPos) {
        lastRes = XN_convLayers[fwdPos].forwardPass(lastRes);
    }

    //TODO: flatten lastRes
    lastRes.reshape(XN_totalLabels, 1, 1);

    return this->softmax(lastRes);
}

arma::vec XnorNetwork::softmax(arma::mat z) {
    vec Y = arma::exp( arma::tanh(z) );
    return Y / arma::accu(Y);
}
