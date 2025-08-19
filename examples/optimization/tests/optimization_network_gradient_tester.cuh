#pragma once

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <random>
#include <string>
#include <tuple>
#include "../../../include/variable.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

namespace xyz_autodiff {
namespace optimization {
namespace test {

// Gradient computation tags
enum class GradientTag {
    Analytical,
    Numerical
};

// Base structure for optimization parameters
struct OptimizationParameters {
    // Model parameters: y = (x1 - a)^2 + b(x2 - c)^2 + d
    float a;        // parameter a
    float b;        // parameter b  
    float c;        // parameter c
    float d;        // parameter d
    float x1;       // input x1
    float x2;       // input x2
    float y_target; // target value
};

// Extended parameters for more complex models
struct ExtendedOptimizationParameters : OptimizationParameters {
    float e;        // additional parameter e
    float f;        // additional parameter f
    float x3;       // additional input x3
};

// Buffer structure for GPU memory
template <typename ParameterStruct>
struct NetworkTestBuffer {
    ParameterStruct value;    // parameter values
    ParameterStruct diff;     // parameter gradients
};

// Kernel to run network forward and backward pass
template <GradientTag tag, typename NetworkFunction, typename ParameterStruct>
__global__ void run_network_kernel(
    NetworkFunction network,
    NetworkTestBuffer<ParameterStruct>* buffer,
    double delta = 1e-7
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        network.template operator()<tag>(&buffer->value, &buffer->diff, delta);
    }
}

// Kernel to compute numerical gradients
template <typename NetworkFunction, typename ParameterStruct>
__global__ void compute_numerical_gradient_kernel(
    NetworkFunction network,
    NetworkTestBuffer<ParameterStruct>* buffer,
    float* param_ptr,
    float* grad_ptr,
    double delta = 1e-7
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Save original value
        float original_value = *param_ptr;
        
        // Reset gradients
        memset(&buffer->diff, 0, sizeof(ParameterStruct));
        
        // Forward computation
        *param_ptr = original_value + delta;
        network.template operator()<GradientTag::Numerical>(&buffer->value, &buffer->diff, delta);
        float f_plus = *grad_ptr;  // Get the loss gradient
        
        // Backward computation
        memset(&buffer->diff, 0, sizeof(ParameterStruct));
        *param_ptr = original_value - delta;
        network.template operator()<GradientTag::Numerical>(&buffer->value, &buffer->diff, delta);
        float f_minus = *grad_ptr;  // Get the loss gradient
        
        // Compute numerical gradient
        *grad_ptr = (f_plus - f_minus) / (2.0 * delta);
        
        // Restore original value
        *param_ptr = original_value;
    }
}

// Main NetworkGradientTester class
template <typename ParameterStruct, typename NetworkFunction>
class NetworkGradientTester {
public:
    // Test single parameter configuration
    static std::tuple<bool, double, std::string> test_single_case(
        NetworkFunction network,
        const ParameterStruct& initial_params,
        double tolerance = 1e-5,
        double delta = 1e-7,
        int seed = 42
    ) {
        // Allocate device memory
        auto device_buffer = makeCudaUnique<NetworkTestBuffer<ParameterStruct>>();
        
        // Initialize buffer with initial parameters
        NetworkTestBuffer<ParameterStruct> host_buffer;
        host_buffer.value = initial_params;
        memset(&host_buffer.diff, 0, sizeof(ParameterStruct));
        
        // Copy to device
        cudaMemcpy(device_buffer.get(), &host_buffer, 
                   sizeof(NetworkTestBuffer<ParameterStruct>), cudaMemcpyHostToDevice);
        
        // Run analytical gradient computation
        run_network_kernel<GradientTag::Analytical><<<1, 1>>>(
            network, device_buffer.get(), delta);
        cudaDeviceSynchronize();
        
        // Copy analytical gradients back
        ParameterStruct analytical_grads;
        cudaMemcpy(&analytical_grads, &device_buffer.get()->diff, 
                   sizeof(ParameterStruct), cudaMemcpyDeviceToHost);
        
        // Reset gradients
        memset(&host_buffer.diff, 0, sizeof(ParameterStruct));
        cudaMemcpy(&device_buffer.get()->diff, &host_buffer.diff, 
                   sizeof(ParameterStruct), cudaMemcpyHostToDevice);
        
        // Run numerical gradient computation
        run_network_kernel<GradientTag::Numerical><<<1, 1>>>(
            network, device_buffer.get(), delta);
        cudaDeviceSynchronize();
        
        // Copy numerical gradients back
        ParameterStruct numerical_grads;
        cudaMemcpy(&numerical_grads, &device_buffer.get()->diff, 
                   sizeof(ParameterStruct), cudaMemcpyDeviceToHost);
        
        // Compare gradients
        double max_error = 0.0;
        std::string error_details = "";
        bool passed = true;
        
        // Check each parameter gradient
        float* analytical_ptr = reinterpret_cast<float*>(&analytical_grads);
        float* numerical_ptr = reinterpret_cast<float*>(&numerical_grads);
        int num_params = sizeof(ParameterStruct) / sizeof(float);
        
        for (int i = 0; i < num_params; ++i) {
            double error = std::abs(analytical_ptr[i] - numerical_ptr[i]);
            double relative_error = error / (std::abs(numerical_ptr[i]) + 1e-10);
            
            if (error > tolerance && relative_error > tolerance) {
                passed = false;
                if (error > max_error) {
                    max_error = error;
                    error_details = "Parameter " + std::to_string(i) + 
                                  ": analytical=" + std::to_string(analytical_ptr[i]) +
                                  ", numerical=" + std::to_string(numerical_ptr[i]) +
                                  ", error=" + std::to_string(error);
                }
            }
        }
        
        return std::make_tuple(passed, max_error, error_details);
    }
    
    // Test network with multiple random configurations
    static void test_network(
        const std::string& network_name,
        NetworkFunction network,
        const ParameterStruct& initial_params,
        int num_tests = 100,
        double tolerance = 1e-5,
        double delta = 1e-7,
        double parameter_range = 0.5
    ) {
        std::cout << "\n=== Testing " << network_name << " ===" << std::endl;
        std::cout << "Number of tests: " << num_tests << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Delta: " << delta << std::endl;
        std::cout << "Parameter range: " << parameter_range << std::endl;
        
        int passed_tests = 0;
        int failed_tests = 0;
        double max_overall_error = 0.0;
        std::string worst_case_details = "";
        
        // Random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-parameter_range, parameter_range);
        
        for (int test_idx = 0; test_idx < num_tests; ++test_idx) {
            // Create random parameter perturbation
            ParameterStruct test_params = initial_params;
            float* param_ptr = reinterpret_cast<float*>(&test_params);
            int num_params = sizeof(ParameterStruct) / sizeof(float);
            
            for (int i = 0; i < num_params; ++i) {
                param_ptr[i] += dist(gen);
            }
            
            // Test this configuration
            auto [passed, error, details] = test_single_case(
                network, test_params, tolerance, delta, test_idx);
            
            if (passed) {
                passed_tests++;
            } else {
                failed_tests++;
                if (error > max_overall_error) {
                    max_overall_error = error;
                    worst_case_details = "Test " + std::to_string(test_idx) + ": " + details;
                }
            }
            
            // Progress update
            if ((test_idx + 1) % 10 == 0) {
                std::cout << "Progress: " << (test_idx + 1) << "/" << num_tests 
                          << " (Passed: " << passed_tests << ", Failed: " << failed_tests << ")" 
                          << std::endl;
            }
        }
        
        // Print final results
        std::cout << "\n=== Results for " << network_name << " ===" << std::endl;
        std::cout << "Passed: " << passed_tests << "/" << num_tests << std::endl;
        std::cout << "Failed: " << failed_tests << "/" << num_tests << std::endl;
        std::cout << "Success rate: " << (100.0 * passed_tests / num_tests) << "%" << std::endl;
        
        if (failed_tests > 0) {
            std::cout << "Maximum error: " << max_overall_error << std::endl;
            std::cout << "Worst case: " << worst_case_details << std::endl;
        }
        
        // Assert that all tests passed
        EXPECT_EQ(failed_tests, 0) << "Some gradient tests failed for " << network_name;
    }
};

} // namespace test
} // namespace optimization
} // namespace xyz_autodiff