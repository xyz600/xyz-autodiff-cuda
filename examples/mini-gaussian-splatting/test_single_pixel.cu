#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gaussian_parameters.h"
#include "../../include/variable.cuh"
#include "../../include/operations/binary/add_logic.cuh"
#include "../../include/operations/binary/mul_logic.cuh"
#include "../../include/operations/unary/neg_logic.cuh"
#include "../../include/operations/unary/exp_logic.cuh"
#include "../../include/operations/unary/broadcast.cuh"
#include "operations/covariance_generation.cuh"
#include "operations/mahalanobis_distance.cuh"
#include "../../include/operations/unary/sym_matrix2_inv_logic.cuh"
#include "../../include/util/cuda_unique_ptr.cuh"

using namespace xyz_autodiff;

__global__ void test_single_pixel_kernel(
    const GaussianParams* gaussians,
    GaussianGrads* gradients,
    float* result,
    int num_gaussians
) {
    if (threadIdx.x > 0 || blockIdx.x > 0) return;
    
    // Single pixel at (100, 100)
    float query_point[2] = {100.0f, 100.0f};
    float target_color[3] = {0.5f, 0.3f, 0.7f};
    
    // Test first Gaussian only
    const GaussianParams& gauss = gaussians[0];
    
    // Create Variables with gradient buffers
    VariableRef<2, float> center(const_cast<float*>(gauss.center), gradients[0].center);
    VariableRef<2, float> scale(const_cast<float*>(gauss.scale), gradients[0].scale);
    VariableRef<1, float> rotation(const_cast<float*>(gauss.rotation), gradients[0].rotation);
    VariableRef<3, float> color(const_cast<float*>(gauss.color), gradients[0].color);
    VariableRef<1, float> opacity(const_cast<float*>(gauss.opacity), gradients[0].opacity);
    VariableRef<2, float> query_pt(query_point, nullptr);
    
    // Build computation graph
    auto covariance = op::scale_rotation_to_covariance_3param(scale, rotation);
    auto inv_covariance = op::sym_matrix2_inv(covariance);
    auto mahalanobis_dist_sq = op::mahalanobis_distance_with_center(query_pt, center, inv_covariance);
    auto scaled_distance = mahalanobis_dist_sq * 0.5f;
    auto neg_scaled = op::neg(scaled_distance);
    auto gaussian_value = op::exp(neg_scaled);
    auto weighted_gauss = gaussian_value * opacity;
    auto gauss_broadcast = op::broadcast<3>(weighted_gauss);
    auto weighted_color = color * gauss_broadcast;
    
    // Run forward pass
    weighted_color.run();
    
    // Store result
    result[0] = weighted_color[0];
    result[1] = weighted_color[1];
    result[2] = weighted_color[2];
    
    // Calculate loss gradients
    float loss_r = weighted_color[0] - target_color[0];
    float loss_g = weighted_color[1] - target_color[1]; 
    float loss_b = weighted_color[2] - target_color[2];
    
    // Clear gradients and set loss gradients
    weighted_color.zero_grad();
    weighted_color.add_grad(0, loss_r);
    weighted_color.add_grad(1, loss_g);
    weighted_color.add_grad(2, loss_b);
    
    // Run backward pass
    weighted_color.backward();
}

int main() {
    std::cout << "Testing single pixel Gaussian splatting..." << std::endl;
    
    // Initialize one Gaussian
    GaussianCollection gaussians;
    std::mt19937 rng(42);
    gaussians.initialize_random(256, 256, rng);
    gaussians.upload_to_device();
    
    // Allocate result buffer
    auto device_result = makeCudaUniqueArray<float>(3);
    
    // Run test
    test_single_pixel_kernel<<<1, 1>>>(
        gaussians.device_params.get(),
        gaussians.device_grads.get(),
        device_result.get(),
        1  // Only test first Gaussian
    );
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Copy result back
    std::vector<float> host_result(3);
    err = cudaMemcpy(host_result.data(), device_result.get(), 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Copy error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Download gradients
    gaussians.download_from_device();
    
    std::cout << "Result color: (" << host_result[0] << ", " << host_result[1] << ", " << host_result[2] << ")" << std::endl;
    std::cout << "Center gradient: (" << gaussians.host_grads[0].center[0] << ", " << gaussians.host_grads[0].center[1] << ")" << std::endl;
    std::cout << "Scale gradient: (" << gaussians.host_grads[0].scale[0] << ", " << gaussians.host_grads[0].scale[1] << ")" << std::endl;
    std::cout << "Color gradient: (" << gaussians.host_grads[0].color[0] << ", " << gaussians.host_grads[0].color[1] << ", " << gaussians.host_grads[0].color[2] << ")" << std::endl;
    
    std::cout << "Single pixel test completed successfully!" << std::endl;
    return 0;
}