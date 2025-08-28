#include "training_config.h"
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdlib>

void TrainingConfig::print() const {
    std::cout << "=== Training Configuration ===" << std::endl;
    std::cout << "Learning rates:" << std::endl;
    std::cout << "  Center:     " << lr_center << std::endl;
    std::cout << "  Scale:      " << lr_scale << std::endl;
    std::cout << "  Rotation:   " << lr_rotation << std::endl;
    std::cout << "  Color:      " << lr_color << std::endl;
    std::cout << "  Opacity:    " << lr_opacity << std::endl;
    std::cout << "Training parameters:" << std::endl;
    std::cout << "  Max iterations: " << max_iterations << std::endl;
    std::cout << "  Save interval:  " << save_interval << std::endl;
    std::cout << "Adam optimizer:" << std::endl;
    std::cout << "  Beta1:    " << beta1 << std::endl;
    std::cout << "  Beta2:    " << beta2 << std::endl;
    std::cout << "  Epsilon:  " << epsilon << std::endl;
    std::cout << "Image configuration:" << std::endl;
    std::cout << "  Target image: " << target_image_path << std::endl;
    std::cout << "  Save images:  " << (save_images ? "yes" : "no") << std::endl;
    std::cout << "==============================" << std::endl;
}

TrainingConfig TrainingConfigParser::parse(int argc, char** argv) {
    TrainingConfig config;
    
    if (argc < 6) {
        print_usage(argv[0]);
        throw std::runtime_error("Insufficient arguments");
    }
    
    try {
        int arg_index = 1;
        
        // Parse required learning rates
        config.lr_center = std::atof(argv[arg_index++]);
        config.lr_scale = std::atof(argv[arg_index++]);
        config.lr_rotation = std::atof(argv[arg_index++]);
        config.lr_color = std::atof(argv[arg_index++]);
        config.lr_opacity = std::atof(argv[arg_index++]);
        
        // Parse remaining required arguments
        while (arg_index < argc) {
            std::string arg = argv[arg_index];
            
            if (arg == "--target" && arg_index + 1 < argc) {
                config.target_image_path = argv[++arg_index];
            } else if (arg == "--max-iterations" && arg_index + 1 < argc) {
                config.max_iterations = std::atoi(argv[++arg_index]);
            } else if (arg == "--save-interval" && arg_index + 1 < argc) {
                config.save_interval = std::atoi(argv[++arg_index]);
            } else if (arg == "--no-save-images") {
                config.save_images = false;
            } else if (arg == "--beta1" && arg_index + 1 < argc) {
                config.beta1 = std::atof(argv[++arg_index]);
            } else if (arg == "--beta2" && arg_index + 1 < argc) {
                config.beta2 = std::atof(argv[++arg_index]);
            } else if (arg == "--epsilon" && arg_index + 1 < argc) {
                config.epsilon = std::atof(argv[++arg_index]);
            } else if (arg == "--help") {
                print_usage(argv[0]);
                exit(0);
            } else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                print_usage(argv[0]);
                throw std::runtime_error("Invalid argument");
            }
            arg_index++;
        }
        
        // Validate configuration
        validate_config(config);
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        throw;
    }
    
    return config;
}

void TrainingConfigParser::print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <lr_center> <lr_scale> <lr_rotation> <lr_color> <lr_opacity> [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << "Required Arguments:" << std::endl;
    std::cout << "  lr_center      Learning rate for Gaussian center positions" << std::endl;
    std::cout << "  lr_scale       Learning rate for Gaussian scale parameters" << std::endl;
    std::cout << "  lr_rotation    Learning rate for Gaussian rotation parameters" << std::endl;
    std::cout << "  lr_color       Learning rate for Gaussian color parameters" << std::endl;
    std::cout << "  lr_opacity     Learning rate for Gaussian opacity parameters" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --target PATH              Target image file path (required)" << std::endl;
    std::cout << "  --max-iterations N         Maximum training iterations (default: 500)" << std::endl;
    std::cout << "  --save-interval N          Save image every N iterations (default: 25)" << std::endl;
    std::cout << "  --no-save-images           Disable saving intermediate images" << std::endl;
    std::cout << "  --beta1 VALUE              Adam optimizer beta1 parameter (default: 0.9)" << std::endl;
    std::cout << "  --beta2 VALUE              Adam optimizer beta2 parameter (default: 0.999)" << std::endl;
    std::cout << "  --epsilon VALUE            Adam optimizer epsilon parameter (default: 1e-8)" << std::endl;
    std::cout << "  --help                     Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Example:" << std::endl;
    std::cout << "  " << program_name << " 0.01 0.005 0.02 0.01 0.01 --target data/target.png --max-iterations 1000 --save-interval 50" << std::endl;
}

void TrainingConfigParser::validate_config(const TrainingConfig& config) {
    if (config.lr_center <= 0 || config.lr_scale <= 0 || config.lr_rotation <= 0 || 
        config.lr_color <= 0 || config.lr_opacity <= 0) {
        throw std::runtime_error("All learning rates must be positive");
    }
    
    if (config.max_iterations <= 0) {
        throw std::runtime_error("Max iterations must be positive");
    }
    
    if (config.save_interval <= 0) {
        throw std::runtime_error("Save interval must be positive");
    }
    
    if (config.target_image_path.empty()) {
        throw std::runtime_error("Target image path must be specified with --target");
    }
    
    if (config.beta1 <= 0 || config.beta1 >= 1) {
        throw std::runtime_error("Beta1 must be in range (0, 1)");
    }
    
    if (config.beta2 <= 0 || config.beta2 >= 1) {
        throw std::runtime_error("Beta2 must be in range (0, 1)");
    }
    
    if (config.epsilon <= 0) {
        throw std::runtime_error("Epsilon must be positive");
    }
}