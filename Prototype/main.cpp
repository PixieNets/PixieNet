#include <iostream>
#include <assert.h>

#include "TestArmadillo.h"
#include "TestBinaryMatrix.h"
#include "TestBinaryLayer.h"


int main() {
    TestArmadillo testArma;
    TestBinaryMatrix testBinaryMatrix;
    TestBinaryLayer testBinaryLayer;

    //testArma.runTest();
    testBinaryMatrix.runAllTests();
    testBinaryLayer.runAllTests();

    return 0;
}