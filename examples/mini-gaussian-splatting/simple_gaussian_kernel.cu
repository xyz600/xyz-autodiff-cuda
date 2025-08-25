#include "gaussian_parameters.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace xyz_autodiff;

// Simple forward-only kernel to test basic functionality
__global__ void simple_gaussian_forward_kernel(
    const GaussianParams* gaussians,
    const float* target_image,
    float* output_image,
    float* loss,
    int image_width,
    int image_height,
    int num_gaussians
) {
    int pixel_x = blockIdx.x * blockDim.x + threadIdx.x;
    int pixel_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (pixel_x >= image_width || pixel_y >= image_height) return;
    
    int pixel_idx = pixel_y * image_width + pixel_x;
    
    // Initialize output pixel
    float pixel_color[3] = {0.0f, 0.0f, 0.0f};
    
    // Simple Gaussian evaluation without autodiff for testing
    float query_x = static_cast<float>(pixel_x);
    float query_y = static_cast<float>(pixel_y);
    
    for (int g = 0; g < num_gaussians; g++) {
        const GaussianParams& gauss = gaussians[g];
        
        // Simple 2D Gaussian without rotation for testing
        float dx = query_x - gauss.center[0];
        float dy = query_y - gauss.center[1];
        
        float sx = gauss.scale[0];
        float sy = gauss.scale[1];
        
        // Simple isotropic Gaussian (ignoring rotation for now)
        float dist_sq = (dx*dx) / (sx*sx) + (dy*dy) / (sy*sy);
        float gaussian_val = expf(-0.5f * dist_sq);
        
        // Apply opacity and color
        float contribution = gaussian_val * gauss.opacity[0];
        pixel_color[0] += contribution * gauss.color[0];
        pixel_color[1] += contribution * gauss.color[1];
        pixel_color[2] += contribution * gauss.color[2];
    }
    
    // Store output
    output_image[pixel_idx * 3 + 0] = pixel_color[0];
    output_image[pixel_idx * 3 + 1] = pixel_color[1];
    output_image[pixel_idx * 3 + 2] = pixel_color[2];
    
    // Calculate loss contribution
    int target_idx = pixel_idx * 3;
    float loss_r = pixel_color[0] - target_image[target_idx + 0];
    float loss_g = pixel_color[1] - target_image[target_idx + 1];
    float loss_b = pixel_color[2] - target_image[target_idx + 2];
    
    float pixel_loss = 0.5f * (loss_r * loss_r + loss_g * loss_g + loss_b * loss_b);
    atomicAdd(loss, pixel_loss);
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
    dim3 block_size(16, 16);
    dim3 grid_size(
        (image_width + block_size.x - 1) / block_size.x,
        (image_height + block_size.y - 1) / block_size.y
    );
    
    // Initialize loss to zero
    cudaMemset(device_loss, 0, sizeof(float));
    
    simple_gaussian_forward_kernel<<<grid_size, block_size>>>(
        device_gaussians,
        device_target_image,
        device_output_image,
        device_loss,
        image_width,
        image_height,
        num_gaussians
    );
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Simple kernel error: " << cudaGetErrorString(err) << std::endl;
    }
}