#include <iostream>

#include "TestArmadillo.h"
#include "TestBinaryMatrix.h"


int main() {
    TestArmadillo testArma;
    TestBinaryMatrix testBinaryMatrix;

    testArma.runTest();
    testBinaryMatrix.runAllTests();

    return 0;
}