#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <gtest/gtest.h>
#include "../../include/variable.cuh"
#include "../../include/util/cuda_unique_ptr.cuh"

namespace xyz_autodiff {
namespace test {

// Tag for selecting gradient computation method
enum class GradientTag {
    Analytical,
    Numerical
};

// Template to hold parameter structures for network testing
template <typename ParameterStruct>
struct NetworkTestBuffer {
    ParameterStruct value;
    ParameterStruct diff_numerical;
    ParameterStruct diff_analytical;
};

// Gaussian parameter structure for mini Gaussian splatting
struct GaussianParameters {
    float quaternion[4];
    float position[3];
    float scale[3];
    float color[3];
    float opacity[1];
    float query_point[2];
};

// Network gradient tester for end-to-end testing
template <typename ParameterStruct, typename NetworkFunction>
class NetworkGradientTester {
public:
    using Buffer = NetworkTestBuffer<ParameterStruct>;
    
    // Test network gradients by comparing analytical vs numerical
    static void test_network(
        const std::string& test_name,
        NetworkFunction network_func,
        const ParameterStruct& initial_params,
        int num_tests = 10,
        double tolerance = 1e-5,
        double delta = 1e-7,
        double param_range = 1.0
    ) {
        std::cout << "\n=== NETWORK GRADIENT TEST for " << test_name << " ===" << std::endl;
        
        bool all_passed = true;
        double max_error = 0.0;
        std::string max_error_location;
        
        for (int test_case = 0; test_case < num_tests; ++test_case) {
            // Generate random parameters around initial values
            ParameterStruct test_params = generate_random_params(initial_params, param_range);
            
            // Test gradients for this parameter set
            auto [passed, max_case_error, error_location] = test_single_case(
                network_func, test_params, tolerance, delta, test_case
            );
            
            if (!passed) {
                all_passed = false;
            }
            
            if (max_case_error > max_error) {
                max_error = max_case_error;
                max_error_location = error_location;
            }
        }
        
        // Print summary
        std::cout << "Number of tests: " << num_tests << std::endl;
        std::cout << "Tolerance: " << tolerance << std::endl;
        std::cout << "Delta: " << delta << std::endl;
        std::cout << "Maximum error: " << max_error << std::endl;
        std::cout << "Max error location: " << max_error_location << std::endl;
        
        if (max_error > tolerance) {
            double recommended_tolerance = max_error * 1.1;
            std::cout << "RECOMMENDATION: Use tolerance >= " << recommended_tolerance 
                      << " for this network" << std::endl;
        }
        
        std::cout << "=========================================" << std::endl;
        
        ASSERT_TRUE(all_passed) << "Network gradient test failed for " << test_name;
    }

private:
    // Test single parameter configuration
    static std::tuple<bool, double, std::string> test_single_case(
        NetworkFunction network_func,
        const ParameterStruct& test_params,
        double tolerance,
        double delta,
        int test_case
    ) {
        // Allocate device buffers
        auto analytical_buffer = makeCudaUnique<Buffer>();
        auto numerical_buffer = makeCudaUnique<Buffer>();
        
        // Initialize host buffers
        Buffer host_analytical = {};
        Buffer host_numerical = {};
        
        // Set initial parameters
        host_analytical.value = test_params;
        host_numerical.value = test_params;
        
        // Copy to device
        cudaMemcpy(analytical_buffer.get(), &host_analytical, sizeof(Buffer), cudaMemcpyHostToDevice);
        cudaMemcpy(numerical_buffer.get(), &host_numerical, sizeof(Buffer), cudaMemcpyHostToDevice);
        
        // Run analytical gradient computation
        run_network_kernel<GradientTag::Analytical>
            <<<1, 1>>>(network_func, analytical_buffer.get());
        cudaDeviceSynchronize();
        
        // Run numerical gradient computation
        run_network_kernel<GradientTag::Numerical>
            <<<1, 1>>>(network_func, numerical_buffer.get(), delta);
        cudaDeviceSynchronize();
        
        // Copy results back
        cudaMemcpy(&host_analytical, analytical_buffer.get(), sizeof(Buffer), cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_numerical, numerical_buffer.get(), sizeof(Buffer), cudaMemcpyDeviceToHost);
        
        // Compare gradients
        return compare_gradients(host_analytical.diff_analytical, 
                               host_numerical.diff_numerical, 
                               tolerance, test_case);
    }
    
    // Generate random parameters around initial values
    static ParameterStruct generate_random_params(const ParameterStruct& initial, double range) {
        ParameterStruct params = initial;
        
        // Add small random perturbations to each parameter
        // This would need to be specialized for each parameter structure
        if constexpr (std::is_same_v<ParameterStruct, GaussianParameters>) {
            for (int i = 0; i < 4; ++i) {
                params.quaternion[i] += ((rand() / double(RAND_MAX)) * 2.0 - 1.0) * range * 0.1;
            }
            for (int i = 0; i < 3; ++i) {
                params.position[i] += ((rand() / double(RAND_MAX)) * 2.0 - 1.0) * range;
                params.scale[i] += ((rand() / double(RAND_MAX)) * 2.0 - 1.0) * range * 0.5;
                params.color[i] += ((rand() / double(RAND_MAX)) * 2.0 - 1.0) * range * 0.3;
            }
            params.opacity[0] += ((rand() / double(RAND_MAX)) * 2.0 - 1.0) * range * 0.2;
            for (int i = 0; i < 2; ++i) {
                params.query_point[i] += ((rand() / double(RAND_MAX)) * 2.0 - 1.0) * range;
            }
        }
        
        return params;
    }
    
    // Compare analytical and numerical gradients
    static std::tuple<bool, double, std::string> compare_gradients(
        const ParameterStruct& analytical,
        const ParameterStruct& numerical,
        double tolerance,
        int test_case
    ) {
        bool passed = true;
        double max_error = 0.0;
        std::string max_error_location;
        
        if constexpr (std::is_same_v<ParameterStruct, GaussianParameters>) {
            // Compare quaternion gradients
            for (int i = 0; i < 4; ++i) {
                double error = std::abs(analytical.quaternion[i] - numerical.quaternion[i]);
                if (error > max_error) {
                    max_error = error;
                    max_error_location = "test case " + std::to_string(test_case) + 
                                       ", quaternion[" + std::to_string(i) + "]";
                }
                if (error > tolerance) {
                    passed = false;
                    EXPECT_LE(error, tolerance) 
                        << "Quaternion gradient mismatch at [" << i << "], analytical=" 
                        << analytical.quaternion[i] << ", numerical=" << numerical.quaternion[i];
                }
            }
            
            // Compare position gradients
            for (int i = 0; i < 3; ++i) {
                double error = std::abs(analytical.position[i] - numerical.position[i]);
                if (error > max_error) {
                    max_error = error;
                    max_error_location = "test case " + std::to_string(test_case) + 
                                       ", position[" + std::to_string(i) + "]";
                }
                if (error > tolerance) {
                    passed = false;
                    EXPECT_LE(error, tolerance) 
                        << "Position gradient mismatch at [" << i << "], analytical=" 
                        << analytical.position[i] << ", numerical=" << numerical.position[i];
                }
            }
            
            // Compare scale gradients
            for (int i = 0; i < 3; ++i) {
                double error = std::abs(analytical.scale[i] - numerical.scale[i]);
                if (error > max_error) {
                    max_error = error;
                    max_error_location = "test case " + std::to_string(test_case) + 
                                       ", scale[" + std::to_string(i) + "]";
                }
                if (error > tolerance) {
                    passed = false;
                    EXPECT_LE(error, tolerance) 
                        << "Scale gradient mismatch at [" << i << "], analytical=" 
                        << analytical.scale[i] << ", numerical=" << numerical.scale[i];
                }
            }
            
            // Compare color gradients
            for (int i = 0; i < 3; ++i) {
                double error = std::abs(analytical.color[i] - numerical.color[i]);
                if (error > max_error) {
                    max_error = error;
                    max_error_location = "test case " + std::to_string(test_case) + 
                                       ", color[" + std::to_string(i) + "]";
                }
                if (error > tolerance) {
                    passed = false;
                    EXPECT_LE(error, tolerance) 
                        << "Color gradient mismatch at [" << i << "], analytical=" 
                        << analytical.color[i] << ", numerical=" << numerical.color[i];
                }
            }
            
            // Compare opacity gradient
            double error = std::abs(analytical.opacity[0] - numerical.opacity[0]);
            if (error > max_error) {
                max_error = error;
                max_error_location = "test case " + std::to_string(test_case) + ", opacity[0]";
            }
            if (error > tolerance) {
                passed = false;
                EXPECT_LE(error, tolerance) 
                    << "Opacity gradient mismatch, analytical=" 
                    << analytical.opacity[0] << ", numerical=" << numerical.opacity[0];
            }
            
            // Compare query_point gradients
            for (int i = 0; i < 2; ++i) {
                double error = std::abs(analytical.query_point[i] - numerical.query_point[i]);
                if (error > max_error) {
                    max_error = error;
                    max_error_location = "test case " + std::to_string(test_case) + 
                                       ", query_point[" + std::to_string(i) + "]";
                }
                if (error > tolerance) {
                    passed = false;
                    EXPECT_LE(error, tolerance) 
                        << "Query point gradient mismatch at [" << i << "], analytical=" 
                        << analytical.query_point[i] << ", numerical=" << numerical.query_point[i];
                }
            }
        }
        
        return {passed, max_error, max_error_location};
    }
    
}; // End class NetworkGradientTester

// Kernel launcher template (must be outside class)
template <GradientTag tag, typename NetworkFunction, typename ParameterStruct>
__global__ void run_network_kernel(
    NetworkFunction network_func,
    NetworkTestBuffer<ParameterStruct>* buffer,
    double delta = 1e-7
) {
    if constexpr (tag == GradientTag::Analytical) {
        network_func.template operator()<tag>(
            &buffer->value, &buffer->diff_analytical
        );
    } else {
        network_func.template operator()<tag>(
            &buffer->value, &buffer->diff_numerical, delta
        );
    }
}

} // namespace test
} // namespace xyz_autodiff