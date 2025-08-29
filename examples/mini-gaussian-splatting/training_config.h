#pragma once

#include <string>
#include <vector>

// Configuration structure for Gaussian Splatting training
struct TrainingConfig {
    // Learning rates for different parameter types
    float lr_center = 0.01f;
    float lr_scale = 0.01f;
    float lr_rotation = 0.01f;
    float lr_color = 0.01f;
    float lr_opacity = 0.01f;
    
    // Training parameters
    int max_iterations = 500;
    int save_interval = 25;
    int num_gaussians = 1000;
    
    // Image parameters
    std::string target_image_path;
    bool save_images = true;
    
    // Adam optimizer parameters
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    
    // Display configuration
    void print() const;
};

// Command line argument parser
class TrainingConfigParser {
public:
    // Parse command line arguments
    static TrainingConfig parse(int argc, char** argv);
    
    // Print usage information
    static void print_usage(const char* program_name);
    
private:
    static void validate_config(const TrainingConfig& config);
};