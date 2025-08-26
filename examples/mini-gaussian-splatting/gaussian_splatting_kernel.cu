#include "gaussian_splatting_kernel.cuh"
#include <iostream>

__global__ void gaussian_splatting_kernel(
    const GaussianParams* gaussians,
    GaussianGrads* gradients, 
    const float* target_image,
    PixelOutput* output_image,
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
    PixelOutput& pixel_out = output_image[pixel_idx];
    pixel_out.color[0] = 0.0f;
    pixel_out.color[1] = 0.0f;
    pixel_out.color[2] = 0.0f;
    pixel_out.alpha = 0.0f;
    pixel_out.loss = 0.0f;
    
    // Query point (current pixel position)
    float query_point[2] = {static_cast<float>(pixel_x), static_cast<float>(pixel_y)};
    
    // Accumulate total color from all Gaussians
    Variable<3, float> total_color;
    total_color[0] = 0.0f;
    total_color[1] = 0.0f;
    total_color[2] = 0.0f;
    
    for (int g = 0; g < num_gaussians; g++) {
        const GaussianParams& gauss = gaussians[g];
        
        // Create Variable references for this Gaussian's parameters (no gradients yet)
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
    
    // Set output pixel color for visualization
    pixel_out.color[0] = total_color[0];
    pixel_out.color[1] = total_color[1];
    pixel_out.color[2] = total_color[2];
    
    // Create target image variable (constant)
    int target_idx = pixel_idx * 3;
    Variable<3, float> target_color;
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
        VariableRef<2, float> query_pt(query_point, query_point);
        
        // Build computation graph for this Gaussian
        auto covariance = op::scale_rotation_to_covariance_3param(scale, rotation);
        auto inv_covariance = op::sym_matrix2_inv(covariance);
        auto mahalanobis_dist_sq = op::mahalanobis_distance_with_center(query_pt, center, inv_covariance);
        auto scaled_distance = mahalanobis_dist_sq * 0.5f;
        auto neg_scaled = op::neg(scaled_distance);
        auto gaussian_value = op::exp(neg_scaled);
        auto weighted_gauss = gaussian_value * opacity;
        auto gauss_broadcast = op::broadcast<3>(weighted_gauss);
        auto weighted_color = color * gauss_broadcast;
        
        // Build full L1 loss computation graph for this Gaussian
        auto color_diff = op::sub(weighted_color, target_color);
        auto l1_loss = op::l1_norm(color_diff);
        
        // Run complete forward and backward pass
        l1_loss.run();
    }
}

void launch_gaussian_splatting(
    const GaussianParams* device_gaussians,
    GaussianGrads* device_gradients,
    const float* device_target_image, 
    PixelOutput* device_output_image,
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
    
    std::cout << "Launching kernel with grid: (" << grid_size.x << ", " << grid_size.y 
              << "), block: (" << block_size.x << ", " << block_size.y << ")" << std::endl;
    
    // Launch kernel
    gaussian_splatting_kernel<<<grid_size, block_size>>>(
        device_gaussians,
        device_gradients,
        device_target_image,
        device_output_image,
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
    
    // Wait for kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
}

__global__ void reduce_loss_kernel(const PixelOutput* output, float* total_loss, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float shared_loss[256];
    
    if (idx < num_pixels) {
        shared_loss[threadIdx.x] = output[idx].loss;
    } else {
        shared_loss[threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_loss[threadIdx.x] += shared_loss[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Write block sum to global memory
    if (threadIdx.x == 0) {
        atomicAdd(total_loss, shared_loss[0]);
    }
}

float calculate_total_loss(const PixelOutput* device_output, int image_width, int image_height) {
    int num_pixels = image_width * image_height;
    
    // Allocate device memory for total loss
    float* device_total_loss;
    cudaError_t err = cudaMalloc(&device_total_loss, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate loss memory: " << cudaGetErrorString(err) << std::endl;
        return 0.0f;
    }
    
    // Initialize to zero
    err = cudaMemset(device_total_loss, 0, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "Failed to initialize loss memory: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_total_loss);
        return 0.0f;
    }
    
    // Launch reduction kernel
    dim3 block_size(256);
    dim3 grid_size((num_pixels + block_size.x - 1) / block_size.x);
    
    reduce_loss_kernel<<<grid_size, block_size>>>(device_output, device_total_loss, num_pixels);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Loss reduction kernel error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_total_loss);
        return 0.0f;
    }
    
    // Copy result to host
    float host_total_loss;
    err = cudaMemcpy(&host_total_loss, device_total_loss, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy loss to host: " << cudaGetErrorString(err) << std::endl;
        host_total_loss = 0.0f;
    }
    
    cudaFree(device_total_loss);
    
    return host_total_loss;
}