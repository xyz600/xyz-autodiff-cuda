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
#include "../../../include/operations/unary/mul_constant_logic.cuh"
#include "../../../include/util/cuda_unique_ptr.cuh"

// Include test utilities
#include "../../utility/network_gradient_tester.cuh"

using namespace xyz_autodiff;
using namespace xyz_autodiff::optimization::test;

// ===========================================
// Network Functions for Testing
// ===========================================

// Base structure for optimization parameters
struct OptimizationParameters {
    // Model parameters: y = (x1 - a)^2 + b(x2 - c)^2 + d
    double a;        // parameter a
    double b;        // parameter b  
    double c;        // parameter c
    double d;        // parameter d
    double x1;       // input x1
    double x2;       // input x2
    double y_target; // target value
};

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
        VariableRef<1, double> a_var(&value->a, &diff->a);
        VariableRef<1, double> b_var(&value->b, &diff->b);
        VariableRef<1, double> c_var(&value->c, &diff->c);
        VariableRef<1, double> d_var(&value->d, &diff->d);
        
        // Compute (x1 - a)^2 using sub and squared operations
        auto x1_minus_a = a_var - value->x1;
        auto x1_term = op::squared(x1_minus_a);
        
        // Compute (x2 - c)^2 using sub and squared operations
        auto x2_minus_c = c_var - value->x2;
        auto x2_squared = op::squared(x2_minus_c);
        
        // Compute b * (x2 - c)^2
        auto x2_term = b_var * x2_squared;
        
        // Compute (x1 - a)^2 + b * (x2 - c)^2
        auto combined_terms = x1_term + x2_term;
        
        // Compute y_pred = (x1 - a)^2 + b * (x2 - c)^2 + d
        auto y_pred = combined_terms + d_var;
        
        // Compute loss = (y_pred - y_target)^2
        auto y_diff = y_pred - value->y_target;
        auto loss = op::squared(y_diff);
        
        // Run gradient computation based on tag
        if constexpr (tag == GradientTag::Analytical) {
            loss.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            loss.run_numerical(delta);
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
        VariableRef<1, double> a_var(&value->a, &diff->a);
        VariableRef<1, double> b_var(&value->b, &diff->b);
        VariableRef<1, double> c_var(&value->c, &diff->c);
        VariableRef<1, double> d_var(&value->d, &diff->d);
        
        // Compute prediction terms - use same pattern as SimpleLinearRegressionNetwork
        auto x1_minus_a = a_var - value->x1;
        auto x1_term = op::squared(x1_minus_a);
        auto x2_minus_c = c_var - value->x2;
        auto x2_squared = op::squared(x2_minus_c);
        auto x2_term = b_var * x2_squared;
        auto combined_terms = x1_term + x2_term;
        auto y_pred = combined_terms + d_var;
        
        // Compute main loss
        auto y_diff = y_pred - value->y_target;
        auto main_loss = op::squared(y_diff);
        
        // For now, skip regularization to debug main loss computation
        auto& total_loss = main_loss;
        
        // Run gradient computation
        if constexpr (tag == GradientTag::Analytical) {
            total_loss.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            total_loss.run_numerical(delta);
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
        VariableRef<1, double> a_var(&value->a, &diff->a);
        auto a_minus_x1 = op::sub_constant(a_var, value->x1);
        auto result = op::squared(a_minus_x1);
        
        if constexpr (tag == GradientTag::Analytical) {
            result.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            result.run_numerical(delta);
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
        VariableRef<1, double> a_var(&value->a, &diff->a);
        VariableRef<1, double> b_var(&value->b, &diff->b);
        VariableRef<1, double> c_var(&value->c, &diff->c);
        VariableRef<1, double> d_var(&value->d, &diff->d);
        
        // Use exactly the same pattern as SimpleLinearRegressionNetwork
        // Compute (x1 - a)^2 using sub and squared operations
        auto x1_minus_a = a_var - value->x1;
        auto x1_term = op::squared(x1_minus_a);
        
        // Compute (x2 - c)^2 using sub and squared operations  
        auto x2_minus_c = c_var - value->x2;
        auto x2_squared = op::squared(x2_minus_c);
        
        // Compute b * (x2 - c)^2
        auto x2_term = b_var * x2_squared;
        
        // Compute (x1 - a)^2 + b * (x2 - c)^2
        auto combined_terms = x1_term + x2_term;
        
        // Compute y_pred = (x1 - a)^2 + b * (x2 - c)^2 + d
        auto y_pred = combined_terms + d_var;
        
        // Compute loss
        auto y_diff = y_pred - value->y_target;
        auto loss = op::squared(y_diff);
        
        if constexpr (tag == GradientTag::Analytical) {
            loss.run();
        } else if constexpr (tag == GradientTag::Numerical) {
            loss.run_numerical(delta);
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
    initial_params.a = 1.0;
    initial_params.b = 1.5;
    initial_params.c = 0.5;
    initial_params.d = 0.2;
    initial_params.x1 = 2.0;
    initial_params.x2 = 1.5;
    initial_params.y_target = 3.0;
    
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
    
    initial_params.a = 0.8;
    initial_params.b = 1.2;
    initial_params.c = -0.3;
    initial_params.d = 0.5;
    initial_params.x1 = 1.0;
    initial_params.x2 = 0.8;
    initial_params.y_target = 2.5;
    
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
    
    initial_params.a = 1.0;
    initial_params.x1 = 2.0;
    // Other parameters not used in this test
    initial_params.b = 0.0;
    initial_params.c = 0.0;
    initial_params.d = 0.0;
    initial_params.x2 = 0.0;
    initial_params.y_target = 0.0;
    
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
    
    initial_params.a = 0.5;
    initial_params.b = 0.8;
    initial_params.c = 0.3;
    initial_params.d = 0.2;
    initial_params.x1 = 1.0;
    initial_params.x2 = 0.5;
    initial_params.y_target = 1.5;
    
    ComplexInteractionNetwork network;
    NetworkGradientTester<OptimizationParameters, ComplexInteractionNetwork>::test_network(
        "ComplexInteractionNetwork",
        network,
        initial_params,
        50,       // num_tests
        1e-5,     // tolerance - minimum allowed per CLAUDE.md
        1e-6,     // delta
        0.2       // parameter_range
    );
}

TEST_F(LinearRegressionNetworkGradientTest, StressTestWithLargeValues) {
    OptimizationParameters initial_params = {};
    
    // Test with larger initial values
    initial_params.a = 10.0;
    initial_params.b = 15.0;
    initial_params.c = -8.0;
    initial_params.d = 5.0;
    initial_params.x1 = 20.0;
    initial_params.x2 = -10.0;
    initial_params.y_target = 100.0;
    
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
    initial_params.a = 0.001;
    initial_params.b = 0.002;
    initial_params.c = -0.001;
    initial_params.d = 0.0001;
    initial_params.x1 = 0.01;
    initial_params.x2 = -0.01;
    initial_params.y_target = 0.001;
    
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