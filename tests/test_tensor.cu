#include "xyz_autodiff/tensor.h"

#include <iostream>

int main() {
    std::cout << "Running simple tensor test..." << std::endl;
    
    xyz_autodiff::Tensor t(10);
    
    if (t.size() == 10) {
        std::cout << "✓ Tensor test passed" << std::endl;
        return 0;
    } else {
        std::cout << "✗ Tensor test failed" << std::endl;
        return 1;
    }
}