#include <xyz_autodiff/xyz_autodiff.h>
#include <iostream>
#include <vector>
#include <cmath>

bool test_simple_backward() {
    std::vector<float> data_a = {2.0f, 3.0f};
    std::vector<float> data_b = {4.0f, 5.0f};
    
    xyz_autodiff::Tensor a({2}, true);
    xyz_autodiff::Tensor b({2}, true);
    
    a.from_cpu(data_a.data());
    b.from_cpu(data_b.data());
    
    auto c = a + b;
    auto loss = c.sum();
    
    loss.backward();
    
    std::vector<float> grad_a(2);
    std::vector<float> grad_b(2);
    
    a.to_cpu(grad_a.data());
    b.to_cpu(grad_b.data());
    
    for (int i = 0; i < 2; ++i) {
        if (std::abs(grad_a[i] - 1.0f) > 1e-6f || std::abs(grad_b[i] - 1.0f) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

bool test_chain_rule() {
    std::vector<float> data = {2.0f};
    
    xyz_autodiff::Tensor x({1}, true);
    x.from_cpu(data.data());
    
    auto y = x * x;
    auto z = y * x;
    
    z.backward();
    
    std::vector<float> grad_x(1);
    x.to_cpu(grad_x.data());
    
    float expected_grad = 3.0f * 2.0f * 2.0f;
    
    return std::abs(grad_x[0] - expected_grad) < 1e-6f;
}

int main() {
    xyz_autodiff::initialize();
    
    std::cout << "Running backward pass tests...\n";
    
    bool all_passed = true;
    
    if (test_simple_backward()) {
        std::cout << "✓ Simple backward test passed\n";
    } else {
        std::cout << "✗ Simple backward test failed\n";
        all_passed = false;
    }
    
    if (test_chain_rule()) {
        std::cout << "✓ Chain rule test passed\n";
    } else {
        std::cout << "✗ Chain rule test failed\n";
        all_passed = false;
    }
    
    xyz_autodiff::cleanup();
    
    if (all_passed) {
        std::cout << "All backward pass tests passed!\n";
        return 0;
    } else {
        std::cout << "Some backward pass tests failed!\n";
        return 1;
    }
}