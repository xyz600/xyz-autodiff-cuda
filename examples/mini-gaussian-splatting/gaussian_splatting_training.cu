#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>

#include "image_utils.h"
#include "gaussian_parameters.h"
#include "gaussian_splatting_kernel.cuh"
#include "../../include/util/cuda_unique_ptr.cuh"

//using namespace xyz_autodiff;

class GaussianSplattingTrainer {
private:
    // Image data
    ImageData target_image;
    float* device_target_image;
    PixelOutput* device_output_image;
    
    // Gaussian collection
    GaussianCollection gaussians;
    
    // Training parameters
    float learning_rate;
    int max_iterations;
    int save_interval;
    
    // Random number generator
    std::mt19937 rng;
    
public:
    GaussianSplattingTrainer(float lr = 0.01f, int max_iter = 1000, int save_freq = 50)
        : learning_rate(lr), max_iterations(max_iter), save_interval(save_freq), 
          device_target_image(nullptr), device_output_image(nullptr)
    {
        // Initialize random seed
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    ~GaussianSplattingTrainer() {
        // Clean up device memory
        if (device_target_image) cudaFree(device_target_image);
        if (device_output_image) cudaFree(device_output_image);
    }
    
    bool load_target_image(const std::string& filename) {
        std::cout << "Loading target image: " << filename << std::endl;
        
        if (!load_image(filename, target_image)) {
            std::cout << "Failed to load image, creating test image instead..." << std::endl;
            target_image = create_test_image(256, 256);
        }
        
        std::cout << "Target image: " << target_image.width << "x" << target_image.height 
                  << " (" << target_image.channels << " channels)" << std::endl;
        
        // Allocate device memory for target image
        int image_size = target_image.width * target_image.height * target_image.channels;
        cudaError_t err = cudaMalloc(&device_target_image, image_size * sizeof(float));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate target image memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Copy target image to device
        err = cudaMemcpy(device_target_image, target_image.data.data(),
                         image_size * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy target image to device: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Allocate device memory for output image
        int pixel_count = target_image.width * target_image.height;
        err = cudaMalloc(&device_output_image, pixel_count * sizeof(PixelOutput));
        if (err != cudaSuccess) {
            std::cerr << "Failed to allocate output image memory: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        return true;
    }
    
    void initialize_gaussians() {
        std::cout << "Initializing Gaussians..." << std::endl;
        gaussians.initialize_random(target_image.width, target_image.height, rng);
        gaussians.upload_to_device();
        std::cout << "Gaussian initialization complete." << std::endl;
    }
    
    void save_current_rendering(int iteration) {
        // Download output image from device
        int pixel_count = target_image.width * target_image.height;
        std::vector<PixelOutput> host_output(pixel_count);
        
        cudaError_t err = cudaMemcpy(host_output.data(), device_output_image,
                                     pixel_count * sizeof(PixelOutput), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to download output image: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        // Convert to ImageData format
        ImageData output_image(target_image.width, target_image.height, 3);
        for (int i = 0; i < pixel_count; i++) {
            output_image.data[i * 3 + 0] = host_output[i].color[0];
            output_image.data[i * 3 + 1] = host_output[i].color[1];
            output_image.data[i * 3 + 2] = host_output[i].color[2];
        }
        
        // Save as JPEG
        std::ostringstream filename;
        filename << "output/iteration_" << std::setfill('0') << std::setw(6) << iteration << ".jpg";
        save_image_jpeg(filename.str(), output_image);
    }
    
    void train() {
        std::cout << "\\n=== Starting Gaussian Splatting Training ===" << std::endl;
        std::cout << "Target image: " << target_image.width << "x" << target_image.height << std::endl;
        std::cout << "Gaussians: " << GaussianCollection::NUM_GAUSSIANS << std::endl;
        std::cout << "Learning rate: " << learning_rate << std::endl;
        std::cout << "Max iterations: " << max_iterations << std::endl;
        
        // Create output directory
        const auto status = system("mkdir -p output");
        if (status != 0) {
            std::cerr << "[warning]: failed to make directory";
        }
        
        // Save initial target image
        save_image_jpeg("output/target.jpg", target_image);
        
        for (int iteration = 0; iteration < 1000; iteration++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Clear gradients
            gaussians.zero_gradients();
            gaussians.upload_to_device();
            
            // Run forward and backward pass
            launch_gaussian_splatting(
                gaussians.device_params.get(),
                gaussians.device_grads.get(),
                device_target_image,
                device_output_image,
                target_image.width,
                target_image.height,
                GaussianCollection::NUM_GAUSSIANS
            );

            const auto total_loss = 1.0;

            // Download gradients
            gaussians.download_from_device();
            
            // Apply Adam optimization
            gaussians.adam_step(learning_rate, 0.9f, 0.999f, 1e-8f, iteration + 1);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            // Print progress
            if (iteration % 10 == 0) {
                std::cout << "Iteration " << std::setw(4) << iteration 
                          << " | Loss: " << std::scientific << std::setprecision(6) << total_loss
                          << " | Time: " << duration.count() << "ms" << std::endl;
            }
            
            // Save intermediate results
            if (iteration % save_interval == 0) {
                save_current_rendering(iteration);
            }
        }
        
        // Save final result
        save_current_rendering(max_iterations);
        
        std::cout << "Training completed!" << std::endl;
    }
};

int main(int argc, char** argv) {
    std::cout << "Gaussian Splatting Training with CUDA Automatic Differentiation" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    // Check CUDA availability
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices available!" << std::endl;
        return -1;
    }
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    // Set device
    cudaSetDevice(0);
    
    // Create trainer
    GaussianSplattingTrainer trainer(0.01f, 500, 25);  // lr=0.01, max_iter=500, save every 25 iterations
    
    // Load target image
    std::string image_file = "data/target.png";
    if (argc > 1) {
        image_file = argv[1];
    }
    
    if (!trainer.load_target_image(image_file)) {
        std::cerr << "Failed to load target image" << std::endl;
        return -1;
    }
    
    // Initialize Gaussians
    trainer.initialize_gaussians();
    
    // Start training
    try {
        trainer.train();
    } catch (const std::exception& e) {
        std::cerr << "Training failed with exception: " << e.what() << std::endl;
        return -1;
    }
    
    std::cout << "Program completed successfully!" << std::endl;
    return 0;
}