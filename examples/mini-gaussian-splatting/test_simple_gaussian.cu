#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "gaussian_parameters.h"
#include "image_utils.h"
#include "../../include/util/cuda_unique_ptr.cuh"

// Forward declaration
void launch_simple_gaussian_forward(
    const GaussianParams* device_gaussians,
    const float* device_target_image,
    float* device_output_image,
    float* device_loss,
    int image_width,
    int image_height,
    int num_gaussians
);

int main() {
    std::cout << "Testing simple Gaussian forward pass..." << std::endl;
    
    const int image_width = 64;
    const int image_height = 64;
    const int num_pixels = image_width * image_height;
    
    // Create simple test image (red circle in center)
    ImageData target_image = create_test_image(image_width, image_height);
    
    // Initialize few Gaussians for testing
    GaussianCollection gaussians;
    std::mt19937 rng(42);
    gaussians.initialize_random(image_width, image_height, rng);
    gaussians.upload_to_device();
    
    // Allocate device memory
    auto device_target_image = makeCudaUniqueArray<float>(num_pixels * 3);
    auto device_output_image = makeCudaUniqueArray<float>(num_pixels * 3);
    auto device_loss = makeCudaUnique<float>();
    
    // Copy target image to device
    cudaError_t err = cudaMemcpy(device_target_image.get(), target_image.data.data(),
                                 num_pixels * 3 * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy target image: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Run forward pass
    launch_simple_gaussian_forward(
        gaussians.device_params.get(),
        device_target_image.get(),
        device_output_image.get(),
        device_loss.get(),
        image_width,
        image_height,
        20  // Test with first 20 Gaussians
    );
    
    // Copy results back
    std::vector<float> host_output(num_pixels * 3);
    float host_loss;
    
    err = cudaMemcpy(host_output.data(), device_output_image.get(),
                     num_pixels * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy output: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    err = cudaMemcpy(&host_loss, device_loss.get(), sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "Failed to copy loss: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Check results
    float max_color = 0.0f;
    for (int i = 0; i < num_pixels * 3; i++) {
        max_color = std::max(max_color, host_output[i]);
    }
    
    std::cout << "Forward pass completed successfully!" << std::endl;
    std::cout << "Total loss: " << host_loss << std::endl;
    std::cout << "Max color value: " << max_color << std::endl;
    
    // Save output image for verification
    ImageData output_img(image_width, image_height, 3);
    output_img.data = host_output;
    save_image_jpeg("simple_output.jpg", output_img);
    save_image_jpeg("simple_target.jpg", target_image);
    
    std::cout << "Images saved: simple_output.jpg, simple_target.jpg" << std::endl;
    
    return 0;
}