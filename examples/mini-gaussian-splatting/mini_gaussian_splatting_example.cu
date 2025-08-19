#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include "../../include/variable.cuh"
#include "../../include/util/cuda_unique_ptr.cuh"

// Include all Gaussian splatting operations
#include "operations/matrix_multiplication.cuh"
#include "operations/element_wise_exp.cuh"
#include "operations/norm_operations.cuh"
#include "operations/norm_addition.cuh"
#include "operations/symmetric_matrix.cuh"
#include "operations/mahalanobis_distance.cuh"
#include "operations/covariance_generation.cuh"
#include "operations/element_wise_multiply.cuh"
#include "../../include/operations/quaternion_to_rotation_matrix_logic.cuh"

using namespace xyz_autodiff;

// Structure to hold Gaussian splatting parameters
struct GaussianSplattingBuffers {
    // Input parameters
    float gaussian_center[2];      // 2D center position
    float gaussian_center_grad[2];
    float gaussian_scale[2];       // 2D scale
    float gaussian_scale_grad[2];
    float gaussian_rotation[1];    // rotation angle
    float gaussian_rotation_grad[1];
    float gaussian_color[3];       // RGB color
    float gaussian_color_grad[3];
    float gaussian_opacity[1];     // opacity
    float gaussian_opacity_grad[1];
    
    // Query point
    float query_point[2];
    float query_point_grad[2];
    
    // Intermediate results
    float covariance_3param[3];
    float inv_covariance_3param[3];
    float mahalanobis_dist_sq[1];
    float gaussian_value[1];
    float color_with_opacity[3];
    float final_result[1];  // scalar result combining all operations
};

// Mini Gaussian splatting evaluation kernel
__global__ void mini_gaussian_splatting_kernel(GaussianSplattingBuffers* buffers) {
    // Create Variable references from buffer data
    VariableRef<float, 2> center(buffers->gaussian_center, buffers->gaussian_center_grad);
    VariableRef<float, 2> scale(buffers->gaussian_scale, buffers->gaussian_scale_grad);
    VariableRef<float, 1> rotation(buffers->gaussian_rotation, buffers->gaussian_rotation_grad);
    VariableRef<float, 3> color(buffers->gaussian_color, buffers->gaussian_color_grad);
    VariableRef<float, 1> opacity(buffers->gaussian_opacity, buffers->gaussian_opacity_grad);
    VariableRef<float, 2> query_point(buffers->query_point, buffers->query_point_grad);
    
    // Step 1: Generate covariance matrix from scale and rotation
    auto covariance = op::scale_rotation_to_covariance_3param(scale, rotation);
    covariance.forward();
    
    // Store intermediate result
    for (int i = 0; i < 3; i++) {
        buffers->covariance_3param[i] = covariance[i];
    }
    
    // Step 2: Compute inverse covariance matrix
    auto inv_covariance = op::symmetric_matrix_2x2_inverse(covariance);
    inv_covariance.forward();
    
    // Store intermediate result
    for (int i = 0; i < 3; i++) {
        buffers->inv_covariance_3param[i] = inv_covariance[i];
    }
    
    // Step 3: Compute Mahalanobis distance
    auto mahalanobis_dist_sq = op::mahalanobis_distance_with_center(query_point, center, inv_covariance);
    mahalanobis_dist_sq.forward();
    
    buffers->mahalanobis_dist_sq[0] = mahalanobis_dist_sq[0];
    
    // Step 4: Compute Gaussian value: exp(-0.5 * distance^2)
    auto scaled_distance = op::scalar_multiply(mahalanobis_dist_sq, 0.5f);
    scaled_distance.forward();
    
    auto gaussian_value = op::element_wise_exp_neg(scaled_distance);
    gaussian_value.forward();
    
    buffers->gaussian_value[0] = gaussian_value[0];
    
    // Step 5: Apply opacity to color (scalar multiplication)
    auto color_with_opacity = op::scalar_multiply(color, opacity[0]);
    color_with_opacity.forward();
    
    // Store color with opacity
    for (int i = 0; i < 3; i++) {
        buffers->color_with_opacity[i] = color_with_opacity[i];
    }
    
    // Step 6: Multiply Gaussian value with color (scalar multiplication)
    auto weighted_color = op::scalar_multiply(color_with_opacity, gaussian_value[0]);
    weighted_color.forward();
    
    // Step 7: Compute L1 + L2 norm of the weighted color as final result
    auto final_result = op::l1_plus_l2_norm(weighted_color);
    final_result.forward();
    
    buffers->final_result[0] = final_result[0];
    
    // Step 8: Compute gradients by running backward pass
    final_result.zero_grad();
    final_result.add_grad(0, 1.0f);  // Set upstream gradient to 1.0
    final_result.backward();
}

// Host function to run mini Gaussian splatting example
void run_mini_gaussian_splatting_example() {
    std::cout << "\n=== Mini Gaussian Splatting Example ===" << std::endl;
    
    // Allocate device memory
    auto device_buffers = makeCudaUnique<GaussianSplattingBuffers>();
    
    // Initialize host data
    GaussianSplattingBuffers host_buffers = {};
    
    // Set Gaussian parameters
    host_buffers.gaussian_center[0] = 0.0f;     // center x
    host_buffers.gaussian_center[1] = 0.0f;     // center y
    host_buffers.gaussian_scale[0] = 1.0f;      // scale x
    host_buffers.gaussian_scale[1] = 0.5f;      // scale y
    host_buffers.gaussian_rotation[0] = 0.1f;   // rotation angle (radians)
    host_buffers.gaussian_color[0] = 1.0f;      // red
    host_buffers.gaussian_color[1] = 0.5f;      // green
    host_buffers.gaussian_color[2] = 0.2f;      // blue
    host_buffers.gaussian_opacity[0] = 0.8f;    // opacity
    
    // Set query point
    host_buffers.query_point[0] = 0.5f;         // query x
    host_buffers.query_point[1] = 0.3f;         // query y
    
    // Copy to device
    cudaMemcpy(device_buffers.get(), &host_buffers, sizeof(GaussianSplattingBuffers), cudaMemcpyHostToDevice);
    
    // Run the kernel
    mini_gaussian_splatting_kernel<<<1, 1>>>(device_buffers.get());
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(&host_buffers, device_buffers.get(), sizeof(GaussianSplattingBuffers), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nGaussian Parameters:" << std::endl;
    std::cout << "  Center: (" << host_buffers.gaussian_center[0] << ", " << host_buffers.gaussian_center[1] << ")" << std::endl;
    std::cout << "  Scale: (" << host_buffers.gaussian_scale[0] << ", " << host_buffers.gaussian_scale[1] << ")" << std::endl;
    std::cout << "  Rotation: " << host_buffers.gaussian_rotation[0] << " radians" << std::endl;
    std::cout << "  Color: (" << host_buffers.gaussian_color[0] << ", " << host_buffers.gaussian_color[1] << ", " << host_buffers.gaussian_color[2] << ")" << std::endl;
    std::cout << "  Opacity: " << host_buffers.gaussian_opacity[0] << std::endl;
    
    std::cout << "\nQuery Point: (" << host_buffers.query_point[0] << ", " << host_buffers.query_point[1] << ")" << std::endl;
    
    std::cout << "\nIntermediate Results:" << std::endl;
    std::cout << "  Covariance (3-param): (" << host_buffers.covariance_3param[0] << ", " << host_buffers.covariance_3param[1] << ", " << host_buffers.covariance_3param[2] << ")" << std::endl;
    std::cout << "  Inverse Covariance: (" << host_buffers.inv_covariance_3param[0] << ", " << host_buffers.inv_covariance_3param[1] << ", " << host_buffers.inv_covariance_3param[2] << ")" << std::endl;
    std::cout << "  Mahalanobis Distance²: " << host_buffers.mahalanobis_dist_sq[0] << std::endl;
    std::cout << "  Gaussian Value: " << host_buffers.gaussian_value[0] << std::endl;
    std::cout << "  Color with Opacity: (" << host_buffers.color_with_opacity[0] << ", " << host_buffers.color_with_opacity[1] << ", " << host_buffers.color_with_opacity[2] << ")" << std::endl;
    
    std::cout << "\nFinal Result (L1+L2 norm): " << host_buffers.final_result[0] << std::endl;
    
    std::cout << "\nGradients:" << std::endl;
    std::cout << "  Center grad: (" << host_buffers.gaussian_center_grad[0] << ", " << host_buffers.gaussian_center_grad[1] << ")" << std::endl;
    std::cout << "  Scale grad: (" << host_buffers.gaussian_scale_grad[0] << ", " << host_buffers.gaussian_scale_grad[1] << ")" << std::endl;
    std::cout << "  Rotation grad: " << host_buffers.gaussian_rotation_grad[0] << std::endl;
    std::cout << "  Color grad: (" << host_buffers.gaussian_color_grad[0] << ", " << host_buffers.gaussian_color_grad[1] << ", " << host_buffers.gaussian_color_grad[2] << ")" << std::endl;
    std::cout << "  Opacity grad: " << host_buffers.gaussian_opacity_grad[0] << std::endl;
    std::cout << "  Query point grad: (" << host_buffers.query_point_grad[0] << ", " << host_buffers.query_point_grad[1] << ")" << std::endl;
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