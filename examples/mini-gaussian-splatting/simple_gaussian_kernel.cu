#include "gaussian_splatting_kernel.cuh"
#include <iostream>

// L1 norm-based forward kernel using automatic differentiation
__global__ void simple_gaussian_forward_kernel(
    const GaussianParams* gaussians,
    const float* target_image,
    float* output_image,
    float* total_loss,
    int image_width,
    int image_height,
    int num_gaussians
) {
    // Calculate pixel coordinates
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixel_x >= image_width || pixel_y >= image_height) return;
    
    int pixel_idx = pixel_y * image_width + pixel_x;
    
    // Initialize output pixel
    output_image[pixel_idx * 3 + 0] = 0.0f;
    output_image[pixel_idx * 3 + 1] = 0.0f;
    output_image[pixel_idx * 3 + 2] = 0.0f;
    
    // Query point (current pixel position)
    float query_point[2] = {static_cast<float>(pixel_x), static_cast<float>(pixel_y)};
    
    // Accumulate total color from all Gaussians
    Variable<3, float> total_color;
    total_color[0] = 0.0f;
    total_color[1] = 0.0f;
    total_color[2] = 0.0f;
    
    for (int g = 0; g < num_gaussians; g++) {
        const GaussianParams& gauss = gaussians[g];
        
        // Create Variable references (no gradients needed for forward pass)
        VariableRef<2, float> center(const_cast<float*>(gauss.center), nullptr);
        VariableRef<2, float> scale(const_cast<float*>(gauss.scale), nullptr);
        VariableRef<1, float> rotation(const_cast<float*>(gauss.rotation), nullptr);
        VariableRef<3, float> color(const_cast<float*>(gauss.color), nullptr);
        VariableRef<1, float> opacity(const_cast<float*>(gauss.opacity), nullptr);
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

        // Run forward pass only
        weighted_color.forward();

        // Accumulate to total color
        total_color[0] += weighted_color[0];
        total_color[1] += weighted_color[1];
        total_color[2] += weighted_color[2];
    }
    
    // Set output pixel color
    output_image[pixel_idx * 3 + 0] = total_color[0];
    output_image[pixel_idx * 3 + 1] = total_color[1];
    output_image[pixel_idx * 3 + 2] = total_color[2];
    
    // Create target image variable (constant)
    int target_idx = pixel_idx * 3;
    Variable<3, float> target_color;
    target_color[0] = target_image[target_idx + 0];
    target_color[1] = target_image[target_idx + 1];
    target_color[2] = target_image[target_idx + 2];
    
    // Compute L1 loss using automatic differentiation
    auto color_diff = op::sub(total_color, target_color);
    auto l1_norm_loss = op::l1_norm(color_diff);
    
    // Run forward pass to compute loss
    l1_norm_loss.forward();
    
    // Accumulate loss using atomicAdd for thread safety
    atomicAdd(total_loss, l1_norm_loss[0]);
}

void launch_simple_gaussian_forward(
    const GaussianParams* device_gaussians,
    const float* device_target_image,
    float* device_output_image,
    float* device_loss,
    int image_width,
    int image_height,
    int num_gaussians
) {
    // Initialize loss to zero
    cudaMemset(device_loss, 0, sizeof(float));
    
    // Calculate grid dimensions
    const int TILE_SIZE = 16;
    dim3 block_size(TILE_SIZE, TILE_SIZE);
    dim3 grid_size(
        (image_width + TILE_SIZE - 1) / TILE_SIZE,
        (image_height + TILE_SIZE - 1) / TILE_SIZE
    );
    
    std::cout << "Launching simple forward kernel with grid: (" << grid_size.x << ", " << grid_size.y 
              << "), block: (" << block_size.x << ", " << block_size.y << ")" << std::endl;
    
    // Launch kernel
    simple_gaussian_forward_kernel<<<grid_size, block_size>>>(
        device_gaussians,
        device_target_image,
        device_output_image,
        device_loss,
        image_width,
        image_height,
        num_gaussians
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Simple forward kernel launch error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Simple forward kernel execution error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}