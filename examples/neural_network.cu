#include <iostream>

#include "xyz_autodiff/tensor.h"

int main() {
    std::cout << "Neural Network example:" << std::endl;
    
    xyz_autodiff::Tensor weights(100);
    xyz_autodiff::Tensor inputs(100);
    xyz_autodiff::Tensor outputs(100);
        
    std::cout << "Neural network computation completed!" << std::endl;
    
    return 0;
}