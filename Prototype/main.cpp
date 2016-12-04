#include <iostream>
#include <assert.h>

#include "TestArmadillo.h"
#include "TestBinaryMatrix.h"


int main() {
    TestArmadillo testArma;
    TestBinaryMatrix testBinaryMatrix;

    //testArma.runTest();
    assert(testBinaryMatrix.runAllTests());

    return 0;
}