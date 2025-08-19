#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include "../../../include/variable.cuh"
#include "../../../include/concept/variable.cuh"
#include "../../../include/concept/operation_node.cuh"
#include "../../../include/operations/binary/mul_logic.cuh"
#include "../../../include/operations/binary/add_logic.cuh"
#include "../../../include/operations/binary/sub_logic.cuh"
#include "../../../include/operations/unary/sub_constant_logic.cuh"
#include "../../../include/operations/unary/squared_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

// Include test utilities
#include "../../utility/network_gradient_tester.cuh"

using namespace xyz_autodiff;
using namespace xyz_autodiff::optimization::test;

// ===========================================
// Network Functions for Testing
// ===========================================

// Simple Linear Regression Network
// Implements: loss = (y_pred - y_target)^2
// where y_pred = (x1 - a)^2 + b*(x2 - c)^2 + d
struct SimpleLinearRegressionNetwork {
    template <GradientTag tag>
    __device__ void operator()(
        OptimizationParameters* value,
        OptimizationParameters* diff,
        double delta = 1e-7
    ) const {
        // Create VariableRef for parameters
        VariableRef<float, 1> a_var(&value->a, &diff->a);
        VariableRef<float, 1> b_var(&value->b, &diff->b);
        VariableRef<float, 1> c_var(&value->c, &diff->c);
        VariableRef<float, 1> d_var(&value->d, &diff->d);
        
        // Compute (x1 - a)^2 using sub and squared operations
        auto x1_minus_a = op::sub_constant(a_var, value->x1);
        auto x1_term = squared(x1_minus_a);
        
        // Compute (x2 - c)^2 using sub and squared operations
        auto x2_minus_c = op::sub_constant(c_var, value->x2);
        auto x2_squared = squared(x2_minus_c);
        
        // Compute b * (x2 - c)^2
        auto x2_term = op::mul(b_var, x2_squared);
        
        // Compute (x1 - a)^2 + b * (x2 - c)^2
        auto combined_terms = op::add(x1_term, x2_term);
        
        // Compute y_pred = (x1 - a)^2 + b * (x2 - c)^2 + d
        auto y_pred = op::add(combined_terms, d_var);
        
        // Compute loss = (y_pred - y_target)^2
        auto y_diff = op::sub_constant(y_pred, value->y_target);
        auto loss = squared(y_diff);
        
        // Run gradient computation based on tag
        if constexpr (tag == GradientTag::Analytical) {
            loss.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            loss.run_numerical(static_cast<float>(delta));
        }
    }
};

// Regularized Linear Regression Network
// Adds L2 regularization to parameters
struct RegularizedLinearRegressionNetwork {
    template <GradientTag tag>
    __device__ void operator()(
        OptimizationParameters* value,
        OptimizationParameters* diff,
        double delta = 1e-7
    ) const {
        // Create VariableRef for parameters
        VariableRef<float, 1> a_var(&value->a, &diff->a);
        VariableRef<float, 1> b_var(&value->b, &diff->b);
        VariableRef<float, 1> c_var(&value->c, &diff->c);
        VariableRef<float, 1> d_var(&value->d, &diff->d);
        
        // Compute prediction terms
        auto x1_minus_a = op::sub_constant(a_var, value->x1);
        auto x1_term = squared(x1_minus_a);
        auto x2_minus_c = op::sub_constant(c_var, value->x2);
        auto x2_squared = squared(x2_minus_c);
        auto x2_term = op::mul(b_var, x2_squared);
        auto combined_terms = op::add(x1_term, x2_term);
        auto y_pred = op::add(combined_terms, d_var);
        
        // Compute main loss
        auto y_diff = op::sub_constant(y_pred, value->y_target);
        auto main_loss = squared(y_diff);
        
        // Add L2 regularization: 0.01 * (a^2 + b^2 + c^2 + d^2)
        auto a_reg = op::mul(a_var, a_var);
        auto b_reg = op::mul(b_var, b_var);
        auto c_reg = op::mul(c_var, c_var);
        auto d_reg = op::mul(d_var, d_var);
        
        auto reg1 = op::add(a_reg, b_reg);
        auto reg2 = op::add(c_reg, d_reg);
        auto reg_sum = op::add(reg1, reg2);
        
        // Scale regularization by 0.01
        float scale_data = 0.01f;
        float scale_grad = 0.0f;
        VariableRef<float, 1> scale_var(&scale_data, &scale_grad);
        auto scaled_reg = op::mul(scale_var, reg_sum);
        
        // Total loss = main_loss + regularization
        auto total_loss = op::add(main_loss, scaled_reg);
        
        // Run gradient computation
        if constexpr (tag == GradientTag::Analytical) {
            total_loss.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            total_loss.run_numerical(static_cast<float>(delta));
        }
    }
};

// Test only the sub and square operations combined
struct SubtractSquareOnlyNetwork {
    template <GradientTag tag>
    __device__ void operator()(
        OptimizationParameters* value,
        OptimizationParameters* diff,
        double delta = 1e-7
    ) const {
        // Only test (a - x1)^2 where x1 is treated as a constant
        VariableRef<float, 1> a_var(&value->a, &diff->a);
        auto a_minus_x1 = op::sub_constant(a_var, value->x1);
        auto result = squared(a_minus_x1);
        
        if constexpr (tag == GradientTag::Analytical) {
            result.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            result.run_numerical(static_cast<float>(delta));
        }
    }
};

// Complex interaction network
// Tests parameter interactions and non-linear combinations
struct ComplexInteractionNetwork {
    template <GradientTag tag>
    __device__ void operator()(
        OptimizationParameters* value,
        OptimizationParameters* diff,
        double delta = 1e-7
    ) const {
        VariableRef<float, 1> a_var(&value->a, &diff->a);
        VariableRef<float, 1> b_var(&value->b, &diff->b);
        VariableRef<float, 1> c_var(&value->c, &diff->c);
        VariableRef<float, 1> d_var(&value->d, &diff->d);
        
        // Complex interactions between parameters
        // Term 1: a * b
        auto ab_product = op::mul(a_var, b_var);
        
        // Term 2: d^2 then c + d^2
        auto d_squared = op::mul(d_var, d_var);
        auto cd_diff = op::add(c_var, d_squared);  // c + d^2
        
        // Term 3: b * (a - x1)^2
        auto a_minus_x1 = op::sub_constant(a_var, value->x1);
        auto a_x1_sq = squared(a_minus_x1);
        auto term3 = op::mul(b_var, a_x1_sq);
        
        // Term 4: c * (b - x2)^2
        auto b_minus_x2 = op::sub_constant(b_var, value->x2);
        auto b_x2_sq = squared(b_minus_x2);
        auto term4 = op::mul(c_var, b_x2_sq);
        
        // Combine all terms
        auto sum1 = op::add(ab_product, cd_diff);
        auto sum2 = op::add(term3, term4);
        auto y_pred = op::add(sum1, sum2);
        
        // Compute loss
        auto y_diff = op::sub_constant(y_pred, value->y_target);
        auto loss = squared(y_diff);
        
        if constexpr (tag == GradientTag::Analytical) {
            loss.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            loss.run_numerical(static_cast<float>(delta));
        }
    }
};

// ===========================================
// Test Class
// ===========================================

class LinearRegressionNetworkGradientTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

// ===========================================
// Test Cases
// ===========================================

TEST_F(LinearRegressionNetworkGradientTest, SimpleLinearRegressionNetwork) {
    OptimizationParameters initial_params = {};
    
    // Initialize with reasonable values
    initial_params.a = 1.0f;
    initial_params.b = 1.5f;
    initial_params.c = 0.5f;
    initial_params.d = 0.2f;
    initial_params.x1 = 2.0f;
    initial_params.x2 = 1.5f;
    initial_params.y_target = 3.0f;
    
    SimpleLinearRegressionNetwork network;
    NetworkGradientTester<OptimizationParameters, SimpleLinearRegressionNetwork>::test_network(
        "SimpleLinearRegressionNetwork",
        network,
        initial_params,
        100,      // num_tests
        1e-5,     // tolerance
        1e-6,     // delta
        0.3       // parameter_range
    );
}

TEST_F(LinearRegressionNetworkGradientTest, RegularizedLinearRegressionNetwork) {
    OptimizationParameters initial_params = {};
    
    initial_params.a = 0.8f;
    initial_params.b = 1.2f;
    initial_params.c = -0.3f;
    initial_params.d = 0.5f;
    initial_params.x1 = 1.0f;
    initial_params.x2 = 0.8f;
    initial_params.y_target = 2.5f;
    
    RegularizedLinearRegressionNetwork network;
    NetworkGradientTester<OptimizationParameters, RegularizedLinearRegressionNetwork>::test_network(
        "RegularizedLinearRegressionNetwork",
        network,
        initial_params,
        50,       // num_tests
        1e-5,     // tolerance
        1e-6,     // delta
        0.2       // parameter_range
    );
}

TEST_F(LinearRegressionNetworkGradientTest, SubtractSquareOperation) {
    OptimizationParameters initial_params = {};
    
    initial_params.a = 1.0f;
    initial_params.x1 = 2.0f;
    // Other parameters not used in this test
    initial_params.b = 0.0f;
    initial_params.c = 0.0f;
    initial_params.d = 0.0f;
    initial_params.x2 = 0.0f;
    initial_params.y_target = 0.0f;
    
    SubtractSquareOnlyNetwork network;
    NetworkGradientTester<OptimizationParameters, SubtractSquareOnlyNetwork>::test_network(
        "SubtractSquareOnlyNetwork",
        network,
        initial_params,
        50,       // num_tests
        1e-5,     // tolerance
        1e-7,     // delta
        0.5       // parameter_range
    );
}

TEST_F(LinearRegressionNetworkGradientTest, ComplexInteractionNetwork) {
    OptimizationParameters initial_params = {};
    
    initial_params.a = 0.5f;
    initial_params.b = 0.8f;
    initial_params.c = 0.3f;
    initial_params.d = 0.2f;
    initial_params.x1 = 1.0f;
    initial_params.x2 = 0.5f;
    initial_params.y_target = 1.5f;
    
    ComplexInteractionNetwork network;
    NetworkGradientTester<OptimizationParameters, ComplexInteractionNetwork>::test_network(
        "ComplexInteractionNetwork",
        network,
        initial_params,
        50,       // num_tests
        1e-4,     // tolerance (relaxed for complex network)
        1e-6,     // delta
        0.2       // parameter_range
    );
}

TEST_F(LinearRegressionNetworkGradientTest, StressTestWithLargeValues) {
    OptimizationParameters initial_params = {};
    
    // Test with larger initial values
    initial_params.a = 10.0f;
    initial_params.b = 15.0f;
    initial_params.c = -8.0f;
    initial_params.d = 5.0f;
    initial_params.x1 = 20.0f;
    initial_params.x2 = -10.0f;
    initial_params.y_target = 100.0f;
    
    SimpleLinearRegressionNetwork network;
    NetworkGradientTester<OptimizationParameters, SimpleLinearRegressionNetwork>::test_network(
        "SimpleLinearRegressionNetwork_LargeValues",
        network,
        initial_params,
        20,       // fewer tests for stress test
        1e-3,     // relaxed tolerance for large values
        1e-5,     // larger delta for numerical stability
        1.0       // larger parameter range
    );
}

TEST_F(LinearRegressionNetworkGradientTest, EdgeCaseNearZero) {
    OptimizationParameters initial_params = {};
    
    // Test with values near zero
    initial_params.a = 0.001f;
    initial_params.b = 0.002f;
    initial_params.c = -0.001f;
    initial_params.d = 0.0001f;
    initial_params.x1 = 0.01f;
    initial_params.x2 = -0.01f;
    initial_params.y_target = 0.001f;
    
    SimpleLinearRegressionNetwork network;
    NetworkGradientTester<OptimizationParameters, SimpleLinearRegressionNetwork>::test_network(
        "SimpleLinearRegressionNetwork_NearZero",
        network,
        initial_params,
        30,       // moderate number of tests
        1e-5,     // standard tolerance
        1e-8,     // smaller delta for small values
        0.001     // very small parameter range
    );
}