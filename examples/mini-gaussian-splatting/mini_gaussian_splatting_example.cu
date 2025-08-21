#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "../../include/variable.cuh"
#include "../../include/util/cuda_unique_ptr.cuh"

// Include all Gaussian splatting operations
#include "../../include/operations/binary/matmul_logic.cuh"
#include "../../include/operations/unary/neg_logic.cuh"
#include "../../include/operations/unary/exp_logic.cuh"
#include "operations/mahalanobis_distance.cuh"
#include "operations/covariance_generation.cuh"
#include "../../include/operations/binary/mul_logic.cuh"
#include "../../include/operations/binary/add_logic.cuh"
#include "../../include/operations/unary/l1_norm_logic.cuh"
#include "../../include/operations/unary/l2_norm_logic.cuh"
// Using standard operations for l1_norm + l2_norm + add
#include "../../include/operations/unary/to_rotation_matrix_logic.cuh"
#include "../../include/operations/unary/broadcast.cuh"
#include "../../include/operations/unary/sym_matrix2_inv_logic.cuh"

using namespace xyz_autodiff;

// Structure to hold Gaussian splatting parameter values
struct GaussianSplattingValues {
    // Input parameters
    float gaussian_center[2];      // 2D center position
    float gaussian_scale[2];       // 2D scale
    float gaussian_rotation[1];    // rotation angle
    float gaussian_color[3];       // RGB color
    float gaussian_opacity[1];     // opacity
    
    // Query point
    float query_point[2];
    
    // Output storage for final result
    float final_result[1];         // scalar result combining all operations
};

// Mini Gaussian splatting evaluation kernel
__global__ void mini_gaussian_splatting_kernel(
    GaussianSplattingValues* values, 
    GaussianSplattingValues* gradients
) {
    // Create Variable references from separate value and gradient buffers
    auto center = VariableRef<2, float>(values->gaussian_center, gradients->gaussian_center);
    VariableRef<2, float> scale(values->gaussian_scale, gradients->gaussian_scale);
    VariableRef<1, float> rotation(values->gaussian_rotation, gradients->gaussian_rotation);
    VariableRef<3, float> color(values->gaussian_color, gradients->gaussian_color);
    VariableRef<1, float> opacity(values->gaussian_opacity, gradients->gaussian_opacity);
    VariableRef<2, float> query_point(values->query_point, gradients->query_point);
    
    // Step 1: Generate covariance matrix from scale and rotation
    auto covariance = op::scale_rotation_to_covariance_3param(scale, rotation);
    
    // Step 2: Compute inverse covariance matrix
    auto inv_covariance = op::sym_matrix2_inv(covariance);
    
    // Step 3: Compute Mahalanobis distance
    auto mahalanobis_dist_sq = op::mahalanobis_distance_with_center(query_point, center, inv_covariance);
    
    // Step 4: Compute Gaussian value: exp(-0.5 * distance^2)
    // Use constant operator for multiplication
    auto scaled_distance = mahalanobis_dist_sq * 0.5f;
    
    // Apply negation and then exponential using standard operations
    auto neg_scaled = op::neg(scaled_distance);
    auto gaussian_value = op::exp(neg_scaled);
    
    // Step 5: Apply opacity to color (element-wise multiplication with opacity broadcast)
    // Use broadcast operation to efficiently broadcast size-1 opacity to size-3
    auto opacity_broadcast = op::broadcast<3>(opacity);
    auto color_with_opacity = color * opacity_broadcast;
    
    // Step 6: Multiply Gaussian value with color
    // Use broadcast operation to efficiently broadcast size-1 gaussian_value to size-3
    auto gauss_broadcast = op::broadcast<3>(gaussian_value);
    auto weighted_color = color_with_opacity * gauss_broadcast;
    
    // Step 7: Compute L1 + L2 norm of the weighted color as final result
    // Use standard operations: l1_norm + l2_norm + add
    auto l1_result = op::l1_norm(weighted_color);
    auto l2_result = op::l2_norm(weighted_color);
    auto final_result_op = l1_result + l2_result;
    
    // Step 8: Compute gradients by running forward and backward pass
    final_result_op.run();
    
    // Store the final result
    values->final_result[0] = final_result_op[0];
}

// Host function to run mini Gaussian splatting example
void run_mini_gaussian_splatting_example() {
    std::cout << "\n=== Mini Gaussian Splatting Example ===" << std::endl;
    
    // Allocate device memory for separate value and gradient structures
    auto device_values = makeCudaUnique<GaussianSplattingValues>();
    auto device_gradients = makeCudaUnique<GaussianSplattingValues>();
    
    // Initialize host data structures
    GaussianSplattingValues host_values = {};
    GaussianSplattingValues host_gradients = {};
    
    // Set Gaussian parameters
    host_values.gaussian_center[0] = 0.0f;     // center x
    host_values.gaussian_center[1] = 0.0f;     // center y
    host_values.gaussian_scale[0] = 1.0f;      // scale x
    host_values.gaussian_scale[1] = 0.5f;      // scale y
    host_values.gaussian_rotation[0] = 0.1f;   // rotation angle (radians)
    host_values.gaussian_color[0] = 1.0f;      // red
    host_values.gaussian_color[1] = 0.5f;      // green
    host_values.gaussian_color[2] = 0.2f;      // blue
    host_values.gaussian_opacity[0] = 0.8f;    // opacity
    
    // Set query point
    host_values.query_point[0] = 0.5f;         // query x
    host_values.query_point[1] = 0.3f;         // query y
    
    // Initialize gradients to zero
    memset(&host_gradients, 0, sizeof(GaussianSplattingValues));
    
    // Copy to device
    cudaMemcpy(device_values.get(), &host_values, sizeof(GaussianSplattingValues), cudaMemcpyHostToDevice);
    cudaMemcpy(device_gradients.get(), &host_gradients, sizeof(GaussianSplattingValues), cudaMemcpyHostToDevice);
    
    // Run the kernel
    mini_gaussian_splatting_kernel<<<1, 1>>>(device_values.get(), device_gradients.get());
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(&host_values, device_values.get(), sizeof(GaussianSplattingValues), cudaMemcpyDeviceToHost);
    cudaMemcpy(&host_gradients, device_gradients.get(), sizeof(GaussianSplattingValues), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nGaussian Parameters:" << std::endl;
    std::cout << "  Center: (" << host_values.gaussian_center[0] << ", " << host_values.gaussian_center[1] << ")" << std::endl;
    std::cout << "  Scale: (" << host_values.gaussian_scale[0] << ", " << host_values.gaussian_scale[1] << ")" << std::endl;
    std::cout << "  Rotation: " << host_values.gaussian_rotation[0] << " radians" << std::endl;
    std::cout << "  Color: (" << host_values.gaussian_color[0] << ", " << host_values.gaussian_color[1] << ", " << host_values.gaussian_color[2] << ")" << std::endl;
    std::cout << "  Opacity: " << host_values.gaussian_opacity[0] << std::endl;
    
    std::cout << "\nQuery Point: (" << host_values.query_point[0] << ", " << host_values.query_point[1] << ")" << std::endl;
    
    std::cout << "\nFinal Result (L1+L2 norm): " << host_values.final_result[0] << std::endl;
    
    std::cout << "\nGradients:" << std::endl;
    std::cout << "  Center grad: (" << host_gradients.gaussian_center[0] << ", " << host_gradients.gaussian_center[1] << ")" << std::endl;
    std::cout << "  Scale grad: (" << host_gradients.gaussian_scale[0] << ", " << host_gradients.gaussian_scale[1] << ")" << std::endl;
    std::cout << "  Rotation grad: " << host_gradients.gaussian_rotation[0] << std::endl;
    std::cout << "  Color grad: (" << host_gradients.gaussian_color[0] << ", " << host_gradients.gaussian_color[1] << ", " << host_gradients.gaussian_color[2] << ")" << std::endl;
    std::cout << "  Opacity grad: " << host_gradients.gaussian_opacity[0] << std::endl;
    std::cout << "  Query point grad: (" << host_gradients.query_point[0] << ", " << host_gradients.query_point[1] << ")" << std::endl;
}

int main() {
    std::cout << "Mini Gaussian Splatting with CUDA Automatic Differentiation" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices available!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Run the example
    try {
        run_mini_gaussian_splatting_example();
        std::cout << "\nExample completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}