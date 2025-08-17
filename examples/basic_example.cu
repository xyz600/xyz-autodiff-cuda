#include <xyz_autodiff/xyz_autodiff.h>
#include <iostream>
#include <vector>

int main() {
    xyz_autodiff::initialize();
    
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 3.0f, 4.0f, 5.0f};
    
    xyz_autodiff::Tensor a({2, 2}, true);
    xyz_autodiff::Tensor b({2, 2}, true);
    
    a.from_cpu(data_a.data());
    b.from_cpu(data_b.data());
    
    std::cout << "Basic operations example:\n";
    
    auto c = a + b;
    auto d = c * a;
    auto loss = d.sum();
    
    std::cout << "Forward pass completed\n";
    
    loss.backward();
    
    std::cout << "Backward pass completed\n";
    
    std::vector<float> result(4);
    loss.to_cpu(result.data());
    std::cout << "Loss: " << result[0] << std::endl;
    
    std::vector<float> grad_a(4);
    a.to_cpu(grad_a.data());
    std::cout << "Gradient of a: ";
    for (float val : grad_a) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    
    xyz_autodiff::cleanup();
    return 0;
}