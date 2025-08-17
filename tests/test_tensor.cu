#include <xyz_autodiff/xyz_autodiff.h>
#include <iostream>
#include <vector>
#include <cmath>

bool test_tensor_creation() {
    xyz_autodiff::Tensor t({2, 3}, true);
    return t.shape()[0] == 2 && t.shape()[1] == 3 && t.size() == 6 && t.requires_grad();
}

bool test_tensor_data_transfer() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    xyz_autodiff::Tensor t({2, 3}, false);
    
    t.from_cpu(data.data());
    
    std::vector<float> result(6);
    t.to_cpu(result.data());
    
    for (int i = 0; i < 6; ++i) {
        if (std::abs(result[i] - data[i]) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

bool test_tensor_copy() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    xyz_autodiff::Tensor t1({2, 2}, true);
    t1.from_cpu(data.data());
    
    xyz_autodiff::Tensor t2 = t1;
    
    std::vector<float> result1(4), result2(4);
    t1.to_cpu(result1.data());
    t2.to_cpu(result2.data());
    
    for (int i = 0; i < 4; ++i) {
        if (std::abs(result1[i] - result2[i]) > 1e-6f) {
            return false;
        }
    }
    
    return t2.requires_grad() == t1.requires_grad();
}

int main() {
    xyz_autodiff::initialize();
    
    std::cout << "Running tensor tests...\n";
    
    bool all_passed = true;
    
    if (test_tensor_creation()) {
        std::cout << "✓ Tensor creation test passed\n";
    } else {
        std::cout << "✗ Tensor creation test failed\n";
        all_passed = false;
    }
    
    if (test_tensor_data_transfer()) {
        std::cout << "✓ Tensor data transfer test passed\n";
    } else {
        std::cout << "✗ Tensor data transfer test failed\n";
        all_passed = false;
    }
    
    if (test_tensor_copy()) {
        std::cout << "✓ Tensor copy test passed\n";
    } else {
        std::cout << "✗ Tensor copy test failed\n";
        all_passed = false;
    }
    
    xyz_autodiff::cleanup();
    
    if (all_passed) {
        std::cout << "All tensor tests passed!\n";
        return 0;
    } else {
        std::cout << "Some tensor tests failed!\n";
        return 1;
    }
}