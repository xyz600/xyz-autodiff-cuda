#pragma once

#include <vector>
#include <string>
#include <memory>

struct ImageData {
    int width;
    int height;
    int channels;
    std::vector<float> data;  // Normalized to [0, 1]
    
    ImageData() : width(0), height(0), channels(0) {}
    ImageData(int w, int h, int c) : width(w), height(h), channels(c), data(w * h * c, 0.0f) {}
};

// Load image from file (supports PNG, JPEG, etc.)
bool load_image(const std::string& filename, ImageData& image);

// Save image to JPEG file
bool save_image_jpeg(const std::string& filename, const ImageData& image, int quality = 90);

// Create a simple test pattern if no image file is available
ImageData create_test_image(int width, int height);