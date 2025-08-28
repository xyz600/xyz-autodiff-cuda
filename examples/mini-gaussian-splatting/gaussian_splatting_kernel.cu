#include "gaussian_splatting_kernel.cuh"
#include "operations/unary/sigmoid_logic.cuh"
#include "operations/unary/const_array_sub_logic.cuh"
#include "operations/operation.cuh"
#include <iostream>

__global__ void gaussian_splatting_kernel(
    const GaussianParams* gaussians,
    GaussianGrads* gradients, 
    const float* target_image,
    PixelOutput* output_image,
    float* total_loss,
    int image_width,
    int image_height,
    int num_gaussians
) {
    // Calculate pixel coordinates from block and thread indices
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (pixel_x >= image_width || pixel_y >= image_height) return;
    
    int pixel_idx = pixel_y * image_width + pixel_x;
    
    // Initialize output pixel
    PixelOutput pixel_out {};
    
    // Query point (current pixel position)
    const float query_point[2] = {static_cast<float>(pixel_x), static_cast<float>(pixel_y)};
        
    for (int g = 0; g < num_gaussians; g++) {
        const GaussianParams& gauss = gaussians[g];
        
        // Create Variable references for this Gaussian's parameters (no gradients yet)
        VariableRef<2, float> center(const_cast<float*>(gauss.center), nullptr);
        VariableRef<2, float> scale(const_cast<float*>(gauss.scale), nullptr);
        VariableRef<1, float> rotation(const_cast<float*>(gauss.rotation), nullptr);
        VariableRef<3, float> color(const_cast<float*>(gauss.color), nullptr);
        VariableRef<1, float> opacity(const_cast<float*>(gauss.opacity), nullptr);
        
        // Build computation graph
        auto exp_scale = op::exp(scale);
        auto covariance = op::scale_rotation_to_covariance_3param(exp_scale, rotation);
        auto inv_covariance = op::sym_matrix2_inv(covariance);
        auto mahalanobis_dist_sq = op::mahalanobis_distance_with_center(query_point[0], query_point[1], center, inv_covariance);
        if (mahalanobis_dist_sq[0] + mahalanobis_dist_sq[1] > 4.0) {
            continue;
        }
        auto scaled_distance = mahalanobis_dist_sq * 0.5f;
        auto neg_scaled = op::neg(scaled_distance);
        auto gaussian_value = op::exp(neg_scaled);
        auto sig_opacity = op::sigmoid(opacity);
        auto weighted_gauss = gaussian_value * sig_opacity;
        auto gauss_broadcast = op::broadcast<3>(weighted_gauss);
        auto weighted_color = color * gauss_broadcast;

        // Run forward pass only
        weighted_color.forward();

        // Accumulate to total color
        pixel_out[0] += weighted_color[0];
        pixel_out[1] += weighted_color[1];
        pixel_out[2] += weighted_color[2];
    }
    
    // Create target image variable (constant)
    // fix: re-order global memory allignment ?
    int target_idx = pixel_idx * 3;
    float target_color[3];
    target_color[0] = target_image[target_idx + 0];
    target_color[1] = target_image[target_idx + 1];
    target_color[2] = target_image[target_idx + 2];
    
    // Compute gradients for each Gaussian using L1 norm automatic differentiation
    for (int g = 0; g < num_gaussians; g++) {
        const GaussianParams& gauss = gaussians[g];
        GaussianGrads& grads = gradients[g];
        
        // Create Variable references with local gradient buffers
        // Create Variable references for this Gaussian's parameters (no gradients yet)
        VariableRef<2, float> center(const_cast<float*>(gauss.center), grads.center);
        VariableRef<2, float> scale(const_cast<float*>(gauss.scale), grads.scale);
        VariableRef<1, float> rotation(const_cast<float*>(gauss.rotation), grads.rotation);
        VariableRef<3, float> color(const_cast<float*>(gauss.color), grads.color);
        VariableRef<1, float> opacity(const_cast<float*>(gauss.opacity), grads.opacity);
        
        // Build computation graph for this Gaussian
        auto exp_scale = op::exp(scale);
        auto covariance = op::scale_rotation_to_covariance_3param(exp_scale, rotation);
        auto inv_covariance = op::sym_matrix2_inv(covariance);
        auto mahalanobis_dist_sq = op::mahalanobis_distance_with_center(query_point[0], query_point[1], center, inv_covariance);
        if (mahalanobis_dist_sq[0] + mahalanobis_dist_sq[1] > 4.0) {
            continue;
        }
        auto scaled_distance = mahalanobis_dist_sq * 0.5f;
        auto neg_scaled = op::neg(scaled_distance);
        auto gaussian_value = op::exp(neg_scaled);
        auto sig_opacity = op::sigmoid(opacity);
        auto weighted_gauss = gaussian_value * sig_opacity;
        auto gauss_broadcast = op::broadcast<3>(weighted_gauss);
        auto weighted_color = color * gauss_broadcast;
        
        PixelOutput rest_sum;
        rest_sum[0] = target_color[0] - pixel_out[0];
        rest_sum[1] = target_color[1] - pixel_out[1];
        rest_sum[2] = target_color[2] - pixel_out[2];

        // Build full L1 loss computation graph for this Gaussian
        auto color_diff = weighted_color - rest_sum;
        auto l1_loss = op::l1_norm(color_diff);
        
        // Run complete forward and backward pass
        l1_loss.run();
        
        // Accumulate this Gaussian's L1 loss to the global total using atomic add
        atomicAdd(total_loss, l1_loss[0]);
    }
}

void launch_gaussian_splatting(
    const GaussianParams* device_gaussians,
    GaussianGrads* device_gradients,
    const float* device_target_image, 
    PixelOutput* device_output_image,
    float* device_total_loss,
    int image_width,
    int image_height,
    int num_gaussians
) {
    // Calculate grid dimensions for 16x16 tiles
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (image_width + TILE_SIZE - 1) / TILE_SIZE,
        (image_height + TILE_SIZE - 1) / TILE_SIZE
    );
    
    // Launch kernel
    gaussian_splatting_kernel<<<grid_size, block_size>>>(
        device_gaussians,
        device_gradients,
        device_target_image,
        device_output_image,
        device_total_loss,
        image_width,
        image_height,
        num_gaussians
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}
