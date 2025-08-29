#include <iostream>
#include <iomanip>
#include <sstream>
#include <random>
#include <chrono>

#include "image_utils.h"
#include "gaussian_parameters.h"
#include "gaussian_splatting_kernel.cuh"
#include "training_config.h"
#include "../../include/util/cuda_unique_ptr.cuh"

//using namespace xyz_autodiff;

class GaussianSplattingTrainer {
private:
    // Image data
    ImageData target_image;
    cuda_unique_ptr<PixelOutput[]> device_target_image;
    cuda_unique_ptr<PixelOutput[]> device_output_image;
    cuda_unique_ptr<float> device_total_loss;
    
    // Training configuration (must be initialized first)
    TrainingConfig config;
    
    // Gaussian collection
    GaussianCollection gaussians;
    
    // Random number generator
    std::mt19937 rng;
    
public:
    GaussianSplattingTrainer(const TrainingConfig& cfg)
        : config(cfg), gaussians(cfg)
    {
        // Initialize random seed
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    }
    
    ~GaussianSplattingTrainer() {
        // cuda_unique_ptr handles automatic cleanup
    }
    
    bool load_target_image(const std::string& filename) {
        std::cout << "Loading target image: " << filename << std::endl;
        
        if (!load_image(filename, target_image)) {
            std::cout << "Failed to load image, creating test image instead..." << std::endl;
            target_image = create_test_image(256, 256);
        }
        
        std::cout << "Target image: " << target_image.width << "x" << target_image.height 
                  << " (" << target_image.channels << " channels)" << std::endl;
        
        // Allocate device memory for target image using CUDA unique pointers
        int image_size = target_image.width * target_image.height;
        device_target_image = makeCudaUniqueArray<PixelOutput>(image_size);
        
        // Copy target image to device
        cudaError_t err = cudaMemcpy(device_target_image.get(), target_image.data.data(),
                                    image_size * sizeof(PixelOutput), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            std::cerr << "Failed to copy target image to device: " << cudaGetErrorString(err) << std::endl;
            return false;
        }
        
        // Allocate device memory for output image
        int pixel_count = target_image.width * target_image.height;
        device_output_image = makeCudaUniqueArray<PixelOutput>(pixel_count);
        
        // Allocate device memory for total loss accumulator
        device_total_loss = makeCudaUnique<float>();
        
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
        
        cudaError_t err = cudaMemcpy(host_output.data(), device_output_image.get(),
                                     pixel_count * sizeof(PixelOutput), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "Failed to download output image: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        // Convert to ImageData format
        ImageData output_image(target_image.width, target_image.height, 3);
        for (int i = 0; i < pixel_count; i++) {
            output_image.data[i * 3 + 0] = host_output[i][0];
            output_image.data[i * 3 + 1] = host_output[i][1];
            output_image.data[i * 3 + 2] = host_output[i][2];
        }
        
        // Save as JPEG
        std::ostringstream filename;
        filename << "output/iteration_" << std::setfill('0') << std::setw(6) << iteration << ".jpg";
        save_image_jpeg(filename.str(), output_image);
    }
    
    void train() {
        std::cout << "\\n=== Starting Gaussian Splatting Training ===" << std::endl;
        std::cout << "Target image: " << target_image.width << "x" << target_image.height << std::endl;
        std::cout << "Gaussians: " << gaussians.num_gaussians << std::endl;
        config.print();
        
        // Create output directory if saving images
        if (config.save_images) {
            const auto status = system("mkdir -p output");
            if (status != 0) {
                std::cerr << "[warning]: failed to make directory";
            }
            
            // Save initial target image
            save_image_jpeg("output/target.jpg", target_image);
        }
        
        for (int iteration = 0; iteration < config.max_iterations; iteration++) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // Clear gradients on device (no need to upload parameters again)
            gaussians.zero_gradients_gpu();
            
            // Zero the total loss accumulator
            float zero_loss = 0.0f;
            cudaMemcpy(device_total_loss.get(), &zero_loss, sizeof(float), cudaMemcpyHostToDevice);
            
            // Run forward and backward pass with atomic L1-norm accumulation
            launch_gaussian_splatting(
                gaussians.device_params.get(),
                gaussians.device_grads.get(),
                device_target_image.get(),
                device_output_image.get(),
                device_total_loss.get(),
                target_image.width,
                target_image.height,
                gaussians.num_gaussians
            );

            // Download total loss from device
            float total_loss = 0.0f;
            cudaMemcpy(&total_loss, device_total_loss.get(), sizeof(float), cudaMemcpyDeviceToHost);

            // Apply GPU Adam optimization with individual learning rates
            gaussians.adam_step_gpu_individual(
                config.lr_center, config.lr_scale, config.lr_rotation,
                config.lr_color, config.lr_opacity,
                config.beta1, config.beta2, config.epsilon, iteration + 1
            );
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            const auto average_loss = total_loss / (target_image.height * target_image.width);
            // Print progress
            if (iteration % 10 == 0) {
                std::cout << "Iteration " << std::setw(4) << iteration 
                          << " | average Loss: " << std::scientific << std::setprecision(6) << average_loss
                          << " | Time: " << duration.count() << "ms" << std::endl;
            }
            
            // Save intermediate results
            if (config.save_images && iteration % config.save_interval == 0) {
                save_current_rendering(iteration);
            }
        }
        
        // Save final result
        if (config.save_images) {
            save_current_rendering(config.max_iterations);
        }
        
        std::cout << "Training completed!" << std::endl;
    }
};

int main(int argc, char** argv) {
    std::cout << "Gaussian Splatting Training with CUDA Automatic Differentiation" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    // Parse command line arguments
    TrainingConfig config;
    try {
        config = TrainingConfigParser::parse(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse arguments: " << e.what() << std::endl;
        return -1;
    }
    
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
    GaussianSplattingTrainer trainer(config);
    
    // Load target image
    if (!trainer.load_target_image(config.target_image_path)) {
        std::cerr << "Failed to load target image: " << config.target_image_path << std::endl;
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