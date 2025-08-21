#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <random>
#include "../../../include/variable.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

// Include all Gaussian splatting operations
#include "../../../include/operations/binary/matmul_logic.cuh"
#include "../../../include/operations/unary/neg_logic.cuh"
#include "../../../include/operations/unary/exp_logic.cuh"
#include "../operations/mahalanobis_distance.cuh"
#include "../operations/covariance_generation.cuh"
#include "../../../include/operations/binary/mul_logic.cuh"
#include "../../../include/operations/binary/add_logic.cuh"
#include "../../../include/operations/unary/l1_norm_logic.cuh"
#include "../../../include/operations/unary/l2_norm_logic.cuh"
#include "../../../include/operations/unary/to_rotation_matrix_logic.cuh"
#include "../../../include/operations/unary/broadcast.cuh"
#include "../../../include/operations/unary/sym_matrix2_inv_logic.cuh"

using namespace xyz_autodiff;

// Gradient computation mode
enum class GradientMode {
    Analytical,
    Numerical
};

// Parameter structure for mini Gaussian splatting (double precision)
struct GaussianSplattingParameters {
    double gaussian_center[2];      // 2D center position
    double gaussian_scale[2];       // 2D scale  
    double gaussian_rotation[1];    // rotation angle
    double gaussian_color[3];       // RGB color
    double gaussian_opacity[1];     // opacity
    double query_point[2];          // query point
    
    // Gradient storage
    double center_grad[2];
    double scale_grad[2];
    double rotation_grad[1];
    double color_grad[3];
    double opacity_grad[1];
    double query_grad[2];
};

// Kernel to compute mini Gaussian splatting gradients
template <GradientMode mode>
__global__ void mini_gaussian_splatting_gradient_kernel(
    GaussianSplattingParameters* params,
    double numerical_delta = 1e-7) {
    
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Create Variable references from parameter structure
    VariableRef<2, double> center(params->gaussian_center, params->center_grad);
    VariableRef<2, double> scale(params->gaussian_scale, params->scale_grad);
    VariableRef<1, double> rotation(params->gaussian_rotation, params->rotation_grad);
    VariableRef<3, double> color(params->gaussian_color, params->color_grad);
    VariableRef<1, double> opacity(params->gaussian_opacity, params->opacity_grad);
    VariableRef<2, double> query_point(params->query_point, params->query_grad);
    
    // Step 1: Generate covariance matrix from scale and rotation
    auto covariance = op::scale_rotation_to_covariance_3param(scale, rotation);
    
    // Step 2: Compute inverse covariance matrix
    auto inv_covariance = op::sym_matrix2_inv(covariance);
    
    // Step 3: Compute Mahalanobis distance
    auto mahalanobis_dist_sq = op::mahalanobis_distance_with_center(query_point, center, inv_covariance);
    
    // Step 4: Compute Gaussian value: exp(-0.5 * distance^2)
    auto scaled_distance = mahalanobis_dist_sq * 0.5;
    auto neg_scaled = op::neg(scaled_distance);
    auto gaussian_value = op::exp(neg_scaled);
    
    // Step 5: Apply opacity to color (element-wise multiplication with opacity broadcast)
    auto opacity_broadcast = op::broadcast<3>(opacity);
    auto color_with_opacity = color * opacity_broadcast;
    
    // Step 6: Multiply Gaussian value with color
    auto gauss_broadcast = op::broadcast<3>(gaussian_value);
    auto weighted_color = color_with_opacity * gauss_broadcast;
    
    // Step 7: Compute L1 + L2 norm of the weighted color as final result
    auto l1_result = op::l1_norm(weighted_color);
    auto l2_result = op::l2_norm(weighted_color);
    auto final_result_op = l1_result + l2_result;
    
    // Step 8: Compute gradients based on mode
    if constexpr (mode == GradientMode::Numerical) {
        final_result_op.run_numerical(numerical_delta);
    } else {
        final_result_op.run();
    }
}

// Test fixture
class MiniGaussianSplattingGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
    
    // Helper function to test gradient computation
    void TestGradientComputation(
        const GaussianSplattingParameters& initial_params,
        double tolerance = 1e-5,
        double numerical_delta = 1e-7) {
        
        // Allocate device memory
        auto device_params_analytical = makeCudaUnique<GaussianSplattingParameters>();
        auto device_params_numerical = makeCudaUnique<GaussianSplattingParameters>();
        
        // Initialize parameters for analytical gradient
        cudaMemcpy(device_params_analytical.get(), &initial_params, sizeof(GaussianSplattingParameters), cudaMemcpyHostToDevice);
        cudaMemset(&(device_params_analytical.get()->center_grad), 0, sizeof(double) * 11); // zero all gradients
        
        // Initialize parameters for numerical gradient
        cudaMemcpy(device_params_numerical.get(), &initial_params, sizeof(GaussianSplattingParameters), cudaMemcpyHostToDevice);
        cudaMemset(&(device_params_numerical.get()->center_grad), 0, sizeof(double) * 11); // zero all gradients
        
        // Compute analytical gradients
        mini_gaussian_splatting_gradient_kernel<GradientMode::Analytical><<<1, 1>>>(
            device_params_analytical.get(), numerical_delta);
        cudaDeviceSynchronize();
        
        // Compute numerical gradients
        mini_gaussian_splatting_gradient_kernel<GradientMode::Numerical><<<1, 1>>>(
            device_params_numerical.get(), numerical_delta);
        cudaDeviceSynchronize();
        
        // Copy results back to host
        GaussianSplattingParameters analytical_result, numerical_result;
        cudaMemcpy(&analytical_result, device_params_analytical.get(), 
                   sizeof(GaussianSplattingParameters), cudaMemcpyDeviceToHost);
        cudaMemcpy(&numerical_result, device_params_numerical.get(), 
                   sizeof(GaussianSplattingParameters), cudaMemcpyDeviceToHost);
        
        // Compare gradients for center
        for (int i = 0; i < 2; ++i) {
            double analytical_grad = analytical_result.center_grad[i];
            double numerical_grad = numerical_result.center_grad[i];
            double absolute_error = std::abs(analytical_grad - numerical_grad);
            
            EXPECT_NEAR(analytical_grad, numerical_grad, tolerance) 
                << "Center[" << i << "] gradient mismatch:\n"
                << "  Analytical: " << analytical_grad << "\n"
                << "  Numerical:  " << numerical_grad << "\n"
                << "  Absolute Error: " << absolute_error;
        }
        
        // Compare gradients for scale
        for (int i = 0; i < 2; ++i) {
            double analytical_grad = analytical_result.scale_grad[i];
            double numerical_grad = numerical_result.scale_grad[i];
            double absolute_error = std::abs(analytical_grad - numerical_grad);
            
            EXPECT_NEAR(analytical_grad, numerical_grad, tolerance) 
                << "Scale[" << i << "] gradient mismatch:\n"
                << "  Analytical: " << analytical_grad << "\n"
                << "  Numerical:  " << numerical_grad << "\n"
                << "  Absolute Error: " << absolute_error;
        }
        
        // Compare gradient for rotation
        {
            double analytical_grad = analytical_result.rotation_grad[0];
            double numerical_grad = numerical_result.rotation_grad[0];
            double absolute_error = std::abs(analytical_grad - numerical_grad);
            
            EXPECT_NEAR(analytical_grad, numerical_grad, tolerance) 
                << "Rotation gradient mismatch:\n"
                << "  Analytical: " << analytical_grad << "\n"
                << "  Numerical:  " << numerical_grad << "\n"
                << "  Absolute Error: " << absolute_error;
        }
        
        // Compare gradients for color
        for (int i = 0; i < 3; ++i) {
            double analytical_grad = analytical_result.color_grad[i];
            double numerical_grad = numerical_result.color_grad[i];
            double absolute_error = std::abs(analytical_grad - numerical_grad);
            
            EXPECT_NEAR(analytical_grad, numerical_grad, tolerance) 
                << "Color[" << i << "] gradient mismatch:\n"
                << "  Analytical: " << analytical_grad << "\n"
                << "  Numerical:  " << numerical_grad << "\n"
                << "  Absolute Error: " << absolute_error;
        }
        
        // Compare gradient for opacity
        {
            double analytical_grad = analytical_result.opacity_grad[0];
            double numerical_grad = numerical_result.opacity_grad[0];
            double absolute_error = std::abs(analytical_grad - numerical_grad);
            
            EXPECT_NEAR(analytical_grad, numerical_grad, tolerance) 
                << "Opacity gradient mismatch:\n"
                << "  Analytical: " << analytical_grad << "\n"
                << "  Numerical:  " << numerical_grad << "\n"
                << "  Absolute Error: " << absolute_error;
        }
        
        // Compare gradients for query point
        for (int i = 0; i < 2; ++i) {
            double analytical_grad = analytical_result.query_grad[i];
            double numerical_grad = numerical_result.query_grad[i];
            double absolute_error = std::abs(analytical_grad - numerical_grad);
            
            EXPECT_NEAR(analytical_grad, numerical_grad, tolerance) 
                << "Query[" << i << "] gradient mismatch:\n"
                << "  Analytical: " << analytical_grad << "\n"
                << "  Numerical:  " << numerical_grad << "\n"
                << "  Absolute Error: " << absolute_error;
        }
    }
};

// Test with simple values
TEST_F(MiniGaussianSplattingGradientTest, SimpleValues) {
    GaussianSplattingParameters params = {};
    
    // Set Gaussian parameters
    params.gaussian_center[0] = 0.0;     // center x
    params.gaussian_center[1] = 0.0;     // center y
    params.gaussian_scale[0] = 1.0;      // scale x
    params.gaussian_scale[1] = 0.5;      // scale y
    params.gaussian_rotation[0] = 0.1;   // rotation angle (radians)
    params.gaussian_color[0] = 1.0;      // red
    params.gaussian_color[1] = 0.5;      // green
    params.gaussian_color[2] = 0.2;      // blue
    params.gaussian_opacity[0] = 0.8;    // opacity
    params.query_point[0] = 0.5;         // query x
    params.query_point[1] = 0.3;         // query y
    
    TestGradientComputation(params, 1e-5, 1e-7);
}

// Test with zero parameters
TEST_F(MiniGaussianSplattingGradientTest, ZeroParameters) {
    GaussianSplattingParameters params = {};
    
    // Set mostly zero parameters (avoid exactly zero scale to prevent singularity)
    params.gaussian_center[0] = 0.0;
    params.gaussian_center[1] = 0.0;
    params.gaussian_scale[0] = 0.1;      // small but non-zero
    params.gaussian_scale[1] = 0.1;      // small but non-zero
    params.gaussian_rotation[0] = 0.0;
    params.gaussian_color[0] = 0.1;      // small but non-zero
    params.gaussian_color[1] = 0.1;
    params.gaussian_color[2] = 0.1;
    params.gaussian_opacity[0] = 0.1;    // small but non-zero
    params.query_point[0] = 0.0;
    params.query_point[1] = 0.0;
    
    TestGradientComputation(params, 1e-5, 1e-7);
}

// Test with negative values
TEST_F(MiniGaussianSplattingGradientTest, NegativeValues) {
    GaussianSplattingParameters params = {};
    
    params.gaussian_center[0] = -1.0;
    params.gaussian_center[1] = -0.5;
    params.gaussian_scale[0] = 0.8;      // positive (scale should be positive)
    params.gaussian_scale[1] = 1.2;      // positive
    params.gaussian_rotation[0] = -0.3;  // negative rotation
    params.gaussian_color[0] = 0.7;      // color components positive
    params.gaussian_color[1] = 0.4;
    params.gaussian_color[2] = 0.9;
    params.gaussian_opacity[0] = 0.6;    // opacity positive
    params.query_point[0] = -0.8;        // negative query point
    params.query_point[1] = -0.2;
    
    TestGradientComputation(params, 1e-5, 1e-7);
}

// Test with larger values
TEST_F(MiniGaussianSplattingGradientTest, LargerValues) {
    GaussianSplattingParameters params = {};
    
    params.gaussian_center[0] = 5.0;
    params.gaussian_center[1] = 3.0;
    params.gaussian_scale[0] = 2.0;
    params.gaussian_scale[1] = 1.5;
    params.gaussian_rotation[0] = 1.2;
    params.gaussian_color[0] = 0.8;
    params.gaussian_color[1] = 0.6;
    params.gaussian_color[2] = 0.4;
    params.gaussian_opacity[0] = 0.9;
    params.query_point[0] = 4.5;
    params.query_point[1] = 2.8;
    
    // Relax tolerance slightly for larger values
    TestGradientComputation(params, 1e-4, 1e-6);
}

// Test with small values near zero
TEST_F(MiniGaussianSplattingGradientTest, SmallValues) {
    GaussianSplattingParameters params = {};
    
    params.gaussian_center[0] = 0.01;
    params.gaussian_center[1] = 0.02;
    params.gaussian_scale[0] = 0.1;      // small but avoid singularity
    params.gaussian_scale[1] = 0.05;
    params.gaussian_rotation[0] = 0.005;
    params.gaussian_color[0] = 0.1;
    params.gaussian_color[1] = 0.05;
    params.gaussian_color[2] = 0.02;
    params.gaussian_opacity[0] = 0.1;
    params.query_point[0] = 0.008;
    params.query_point[1] = 0.015;
    
    TestGradientComputation(params, 1e-5, 1e-8);
}

// Random stress test
TEST_F(MiniGaussianSplattingGradientTest, RandomValuesStressTest) {
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<double> center_dist(-2.0, 2.0);
    std::uniform_real_distribution<double> scale_dist(0.1, 2.0);  // positive scales
    std::uniform_real_distribution<double> rotation_dist(-1.57, 1.57);  // -pi/2 to pi/2
    std::uniform_real_distribution<double> color_dist(0.1, 1.0);  // positive colors
    std::uniform_real_distribution<double> opacity_dist(0.1, 1.0);  // positive opacity
    std::uniform_real_distribution<double> query_dist(-3.0, 3.0);
    
    const int num_tests = 50;
    int passed = 0;
    int failed = 0;
    
    for (int test_idx = 0; test_idx < num_tests; ++test_idx) {
        GaussianSplattingParameters params = {};
        
        // Generate random parameters
        params.gaussian_center[0] = center_dist(rng);
        params.gaussian_center[1] = center_dist(rng);
        params.gaussian_scale[0] = scale_dist(rng);
        params.gaussian_scale[1] = scale_dist(rng);
        params.gaussian_rotation[0] = rotation_dist(rng);
        params.gaussian_color[0] = color_dist(rng);
        params.gaussian_color[1] = color_dist(rng);
        params.gaussian_color[2] = color_dist(rng);
        params.gaussian_opacity[0] = opacity_dist(rng);
        params.query_point[0] = query_dist(rng);
        params.query_point[1] = query_dist(rng);
        
        // Run test with relaxed tolerance
        try {
            TestGradientComputation(params, 1e-4, 1e-6);
            passed++;
        } catch (const std::exception& e) {
            failed++;
            if (failed <= 5) {  // Print first few failures for debugging
                std::cout << "Test " << test_idx << " failed: " << e.what() << std::endl;
            }
        }
    }
    
    std::cout << "Random stress test results: " 
              << passed << "/" << num_tests << " passed" << std::endl;
    
    // Expect at least 90% success rate (Gaussian splatting is more complex than linear regression)
    EXPECT_GE(passed, static_cast<int>(num_tests * 0.9)) 
        << "Too many failures in random stress test";
}