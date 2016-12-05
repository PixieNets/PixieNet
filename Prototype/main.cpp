#include <iostream>
#include <assert.h>

#include "TestArmadillo.h"
#include "TestBinaryMatrix.h"
#include "TestBinaryLayer.h"
#include "TestBinaryTensor3D.h"


int main() {
    TestArmadillo testArma;
    TestBinaryMatrix testBinaryMatrix;
    TestBinaryLayer testBinaryLayer;
    TestBinaryTensor3D testBinaryTensor3D;

    //testArma.runTest();
    testBinaryMatrix.runAllTests();
    testBinaryLayer.runAllTests();
    testBinaryTensor3D.runAllTests();

    return 0;
}