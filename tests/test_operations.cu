#include <xyz_autodiff/xyz_autodiff.h>
#include <iostream>
#include <vector>
#include <cmath>

bool test_addition() {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 3.0f, 4.0f, 5.0f};
    
    xyz_autodiff::Tensor a({2, 2}, false);
    xyz_autodiff::Tensor b({2, 2}, false);
    
    a.from_cpu(data_a.data());
    b.from_cpu(data_b.data());
    
    auto c = a + b;
    
    std::vector<float> result(4);
    c.to_cpu(result.data());
    
    std::vector<float> expected = {3.0f, 5.0f, 7.0f, 9.0f};
    
    for (int i = 0; i < 4; ++i) {
        if (std::abs(result[i] - expected[i]) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

bool test_multiplication() {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {2.0f, 2.0f, 2.0f, 2.0f};
    
    xyz_autodiff::Tensor a({2, 2}, false);
    xyz_autodiff::Tensor b({2, 2}, false);
    
    a.from_cpu(data_a.data());
    b.from_cpu(data_b.data());
    
    auto c = a * b;
    
    std::vector<float> result(4);
    c.to_cpu(result.data());
    
    std::vector<float> expected = {2.0f, 4.0f, 6.0f, 8.0f};
    
    for (int i = 0; i < 4; ++i) {
        if (std::abs(result[i] - expected[i]) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

bool test_relu() {
    std::vector<float> data = {-1.0f, 2.0f, -3.0f, 4.0f};
    
    xyz_autodiff::Tensor a({2, 2}, false);
    a.from_cpu(data.data());
    
    auto result_tensor = a.relu();
    
    std::vector<float> result(4);
    result_tensor.to_cpu(result.data());
    
    std::vector<float> expected = {0.0f, 2.0f, 0.0f, 4.0f};
    
    for (int i = 0; i < 4; ++i) {
        if (std::abs(result[i] - expected[i]) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

bool test_matmul() {
    std::vector<float> data_a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> data_b = {1.0f, 0.0f, 0.0f, 1.0f};
    
    xyz_autodiff::Tensor a({2, 2}, false);
    xyz_autodiff::Tensor b({2, 2}, false);
    
    a.from_cpu(data_a.data());
    b.from_cpu(data_b.data());
    
    auto c = a.matmul(b);
    
    std::vector<float> result(4);
    c.to_cpu(result.data());
    
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    
    for (int i = 0; i < 4; ++i) {
        if (std::abs(result[i] - expected[i]) > 1e-6f) {
            return false;
        }
    }
    
    return true;
}

int main() {
    xyz_autodiff::initialize();
    
    std::cout << "Running operations tests...\n";
    
    bool all_passed = true;
    
    if (test_addition()) {
        std::cout << "✓ Addition test passed\n";
    } else {
        std::cout << "✗ Addition test failed\n";
        all_passed = false;
    }
    
    if (test_multiplication()) {
        std::cout << "✓ Multiplication test passed\n";
    } else {
        std::cout << "✗ Multiplication test failed\n";
        all_passed = false;
    }
    
    if (test_relu()) {
        std::cout << "✓ ReLU test passed\n";
    } else {
        std::cout << "✗ ReLU test failed\n";
        all_passed = false;
    }
    
    if (test_matmul()) {
        std::cout << "✓ Matrix multiplication test passed\n";
    } else {
        std::cout << "✗ Matrix multiplication test failed\n";
        all_passed = false;
    }
    
    xyz_autodiff::cleanup();
    
    if (all_passed) {
        std::cout << "All operations tests passed!\n";
        return 0;
    } else {
        std::cout << "Some operations tests failed!\n";
        return 1;
    }
}