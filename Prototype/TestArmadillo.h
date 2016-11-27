//
// Created by Zal on 11/22/16.
//

#ifndef PROTOTYPE_TESTARMADILLO_H
#define PROTOTYPE_TESTARMADILLO_H

#include <armadillo>
using namespace arma;

class TestArmadillo {
public:
    void testGeneral();
    void testSoftmax();

    void runTest(); // runs all tests

};


#endif //PROTOTYPE_TESTARMADILLO_H
