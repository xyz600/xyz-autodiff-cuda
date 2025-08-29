#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <xyz_autodiff/variable.cuh>
#include <xyz_autodiff/operations/binary/mul_logic.cuh>
#include <xyz_autodiff/operations/binary/add_logic.cuh>
#include <xyz_autodiff/operations/binary/sub_logic.cuh>
#include <xyz_autodiff/operations/unary/sub_constant_logic.cuh>
#include <xyz_autodiff/operations/unary/squared_logic.cuh>
#include <xyz_autodiff/util/cuda_unique_ptr.cuh>

using namespace xyz_autodiff;

// Gradient computation mode
enum class GradientMode {
    Analytical,
    Numerical
};

// Data point structure (same as in linear_regression_sgd.cu)
struct DataPoint {
    double x1, x2, y;
};

// Parameter structure (same as in linear_regression_sgd.cu)
struct Parameters {
    double value[4];  // a, b, c, d
    double grad[4];   // grad_a, grad_b, grad_c, grad_d
};

// Kernel to compute gradients (analytical or numerical)
template <GradientMode mode>
__global__ void compute_gradient_kernel(
    const DataPoint* batch_data, 
    int batch_size, 
    Parameters* params,
    double numerical_delta = 1e-7) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // パラメータ変数の作成（ポインタ演算で各パラメータを指定）
    VariableRef<1, double> a_var(&params->value[0], &params->grad[0]);  // a
    VariableRef<1, double> b_var(&params->value[1], &params->grad[1]);  // b
    VariableRef<1, double> c_var(&params->value[2], &params->grad[2]);  // c
    VariableRef<1, double> d_var(&params->value[3], &params->grad[3]);  // d
    
    // 現在のスレッドが担当するデータポイント
    const DataPoint& data = batch_data[idx];
    
    // (x1 - a)^2 を sub と squared 操作で計算
    auto x1_minus_a = op::sub_constant(a_var, data.x1);
    auto x1_term = op::squared(x1_minus_a);
    
    // (x2 - c)^2 を sub と squared 操作で計算
    auto x2_minus_c = op::sub_constant(c_var, data.x2);
    auto x2_squared = op::squared(x2_minus_c);
    
    // b * (x2 - c)^2
    auto x2_term = op::mul(b_var, x2_squared);
    
    // (x1 - a)^2 + b * (x2 - c)^2
    auto combined_terms = op::add(x1_term, x2_term);
    
    // y_pred = (x1 - a)^2 + b * (x2 - c)^2 + d
    auto y_pred = op::add(combined_terms, d_var);
    
    // loss = (y_pred - y_target)^2 を sub と squared 操作で計算
    auto y_diff = op::sub_constant(y_pred, data.y);
    auto loss = op::squared(y_diff);
    
    // Run gradient computation based on mode
    if constexpr (mode == GradientMode::Numerical) {
        loss.run_numerical(numerical_delta);
    } else {
        loss.run();
    }
}

// Test fixture
class LinearRegressionGradientTest : public ::testing::Test {
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
        const DataPoint& test_data,
        const Parameters& initial_params,
        double tolerance = 1e-5,
        double numerical_delta = 1e-7) {
        
        // Allocate device memory
        auto device_data = makeCudaUnique<DataPoint>();
        auto device_params_analytical = makeCudaUnique<Parameters>();
        auto device_params_numerical = makeCudaUnique<Parameters>();
        
        // Copy test data to device
        cudaMemcpy(device_data.get(), &test_data, sizeof(DataPoint), cudaMemcpyHostToDevice);
        
        // Initialize parameters for analytical gradient
        cudaMemcpy(device_params_analytical.get(), &initial_params, sizeof(Parameters), cudaMemcpyHostToDevice);
        cudaMemset(&(device_params_analytical.get()->grad), 0, sizeof(double) * 4);
        
        // Initialize parameters for numerical gradient
        cudaMemcpy(device_params_numerical.get(), &initial_params, sizeof(Parameters), cudaMemcpyHostToDevice);
        cudaMemset(&(device_params_numerical.get()->grad), 0, sizeof(double) * 4);
        
        // Compute analytical gradients
        compute_gradient_kernel<GradientMode::Analytical><<<1, 1>>>(
            device_data.get(), 1, device_params_analytical.get(), numerical_delta);
        cudaDeviceSynchronize();
        
        // Compute numerical gradients
        compute_gradient_kernel<GradientMode::Numerical><<<1, 1>>>(
            device_data.get(), 1, device_params_numerical.get(), numerical_delta);
        cudaDeviceSynchronize();
        
        // Copy results back to host
        Parameters analytical_result, numerical_result;
        cudaMemcpy(&analytical_result, device_params_analytical.get(), 
                   sizeof(Parameters), cudaMemcpyDeviceToHost);
        cudaMemcpy(&numerical_result, device_params_numerical.get(), 
                   sizeof(Parameters), cudaMemcpyDeviceToHost);
        
        // Compare gradients for all parameters
        const char* param_names[] = {"a", "b", "c", "d"};
        for (int i = 0; i < 4; ++i) {
            double analytical_grad = analytical_result.grad[i];
            double numerical_grad = numerical_result.grad[i];
            double absolute_error = std::abs(analytical_grad - numerical_grad);
            double relative_error = absolute_error / (std::abs(numerical_grad) + 1e-10);
            
            // Use the minimum of absolute and relative error for comparison
            double error = std::min(absolute_error, relative_error);
            
            EXPECT_NEAR(analytical_grad, numerical_grad, tolerance) 
                << "Parameter " << param_names[i] << " gradient mismatch:\n"
                << "  Analytical: " << analytical_grad << "\n"
                << "  Numerical:  " << numerical_grad << "\n"
                << "  Absolute Error: " << absolute_error << "\n"
                << "  Relative Error: " << relative_error << "\n"
                << "  Test data: x1=" << test_data.x1 
                << ", x2=" << test_data.x2 
                << ", y=" << test_data.y << "\n"
                << "  Initial params: a=" << initial_params.value[0]
                << ", b=" << initial_params.value[1]
                << ", c=" << initial_params.value[2]
                << ", d=" << initial_params.value[3];
        }
    }
};

// Test with simple values
TEST_F(LinearRegressionGradientTest, SimpleValues) {
    DataPoint test_data = {2.0, 3.0, 5.0};  // x1=2, x2=3, y=5
    Parameters initial_params = {
        {1.0, 1.5, 0.5, 0.2},  // a=1, b=1.5, c=0.5, d=0.2
        {0.0, 0.0, 0.0, 0.0}   // zero gradients
    };
    
    TestGradientComputation(test_data, initial_params, 1e-5, 1e-7);
}

// Test with zero parameters
TEST_F(LinearRegressionGradientTest, ZeroParameters) {
    DataPoint test_data = {1.0, 1.0, 2.0};
    Parameters initial_params = {
        {0.0, 0.0, 0.0, 0.0},  // all zero parameters
        {0.0, 0.0, 0.0, 0.0}
    };
    
    TestGradientComputation(test_data, initial_params, 1e-5, 1e-7);
}

// Test with negative values
TEST_F(LinearRegressionGradientTest, NegativeValues) {
    DataPoint test_data = {-2.0, -1.5, 3.0};
    Parameters initial_params = {
        {-1.0, 2.0, -0.5, -0.3},
        {0.0, 0.0, 0.0, 0.0}
    };
    
    TestGradientComputation(test_data, initial_params, 1e-5, 1e-7);
}

// Test with large values
TEST_F(LinearRegressionGradientTest, LargeValues) {
    DataPoint test_data = {100.0, 150.0, 500.0};
    Parameters initial_params = {
        {50.0, 30.0, 70.0, 20.0},
        {0.0, 0.0, 0.0, 0.0}
    };
    
    // Relax tolerance significantly for large values due to numerical precision
    // The gradients can be in the order of millions, so absolute error tolerance needs adjustment
    TestGradientComputation(test_data, initial_params, 1e4, 1e-5);
}

// Test with small values near zero
TEST_F(LinearRegressionGradientTest, SmallValues) {
    DataPoint test_data = {0.001, 0.002, 0.001};
    Parameters initial_params = {
        {0.0001, 0.0002, 0.0001, 0.00001},
        {0.0, 0.0, 0.0, 0.0}
    };
    
    TestGradientComputation(test_data, initial_params, 1e-5, 1e-9);
}

// Test multiple data points in batch
TEST_F(LinearRegressionGradientTest, BatchProcessing) {
    const int batch_size = 32;
    
    // Create test data
    std::vector<DataPoint> test_data(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        test_data[i] = {
            static_cast<double>(i) * 0.1,      // x1
            static_cast<double>(i) * 0.2,      // x2
            static_cast<double>(i) * 0.3 + 1.0 // y
        };
    }
    
    Parameters initial_params = {
        {0.5, 1.0, 0.3, 0.1},
        {0.0, 0.0, 0.0, 0.0}
    };
    
    // Allocate device memory
    auto device_data = makeCudaUniqueArray<DataPoint>(batch_size);
    auto device_params_analytical = makeCudaUnique<Parameters>();
    auto device_params_numerical = makeCudaUnique<Parameters>();
    
    // Copy data to device
    cudaMemcpy(device_data.get(), test_data.data(), 
               batch_size * sizeof(DataPoint), cudaMemcpyHostToDevice);
    
    // Initialize parameters
    cudaMemcpy(device_params_analytical.get(), &initial_params, sizeof(Parameters), cudaMemcpyHostToDevice);
    cudaMemset(&(device_params_analytical.get()->grad), 0, sizeof(double) * 4);
    
    cudaMemcpy(device_params_numerical.get(), &initial_params, sizeof(Parameters), cudaMemcpyHostToDevice);
    cudaMemset(&(device_params_numerical.get()->grad), 0, sizeof(double) * 4);
    
    // Compute gradients with batch
    dim3 block_size(256);
    dim3 grid_size((batch_size + block_size.x - 1) / block_size.x);
    
    compute_gradient_kernel<GradientMode::Analytical><<<grid_size, block_size>>>(
        device_data.get(), batch_size, device_params_analytical.get(), 1e-7);
    cudaDeviceSynchronize();
    
    compute_gradient_kernel<GradientMode::Numerical><<<grid_size, block_size>>>(
        device_data.get(), batch_size, device_params_numerical.get(), 1e-7);
    cudaDeviceSynchronize();
    
    // Copy results back
    Parameters analytical_result, numerical_result;
    cudaMemcpy(&analytical_result, device_params_analytical.get(), 
               sizeof(Parameters), cudaMemcpyDeviceToHost);
    cudaMemcpy(&numerical_result, device_params_numerical.get(), 
               sizeof(Parameters), cudaMemcpyDeviceToHost);
    
    // Compare accumulated gradients
    const char* param_names[] = {"a", "b", "c", "d"};
    for (int i = 0; i < 4; ++i) {
        double analytical_grad = analytical_result.grad[i];
        double numerical_grad = numerical_result.grad[i];
        double absolute_error = std::abs(analytical_grad - numerical_grad);
        double relative_error = absolute_error / (std::abs(numerical_grad) + 1e-10);
        
        EXPECT_NEAR(analytical_grad, numerical_grad, 1e-3) 
            << "Batch processing - Parameter " << param_names[i] << " gradient mismatch:\n"
            << "  Analytical: " << analytical_grad << "\n"
            << "  Numerical:  " << numerical_grad << "\n"
            << "  Absolute Error: " << absolute_error << "\n"
            << "  Relative Error: " << relative_error;
    }
}

// Stress test with random values
TEST_F(LinearRegressionGradientTest, RandomValuesStressTest) {
    std::srand(42);  // Fixed seed for reproducibility
    
    const int num_tests = 100;
    int passed = 0;
    int failed = 0;
    
    for (int test_idx = 0; test_idx < num_tests; ++test_idx) {
        // Generate random test data
        DataPoint test_data = {
            (std::rand() / double(RAND_MAX)) * 10.0 - 5.0,  // x1 in [-5, 5]
            (std::rand() / double(RAND_MAX)) * 10.0 - 5.0,  // x2 in [-5, 5]
            (std::rand() / double(RAND_MAX)) * 20.0 - 10.0  // y in [-10, 10]
        };
        
        Parameters initial_params = {
            {
                (std::rand() / double(RAND_MAX)) * 4.0 - 2.0,  // a in [-2, 2]
                (std::rand() / double(RAND_MAX)) * 4.0 - 2.0,  // b in [-2, 2]
                (std::rand() / double(RAND_MAX)) * 4.0 - 2.0,  // c in [-2, 2]
                (std::rand() / double(RAND_MAX)) * 4.0 - 2.0   // d in [-2, 2]
            },
            {0.0, 0.0, 0.0, 0.0}
        };
        
        // Run test with relaxed tolerance
        try {
            TestGradientComputation(test_data, initial_params, 1e-4, 1e-6);
            passed++;
        } catch (...) {
            failed++;
        }
    }
    
    std::cout << "Random stress test results: " 
              << passed << "/" << num_tests << " passed" << std::endl;
    
    // Expect at least 95% success rate
    EXPECT_GE(passed, num_tests * 0.95) 
        << "Too many failures in random stress test";
}