#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION  
#include "external/stb/stb_image_write.h"

#include <iostream>
#include <cmath>

bool load_image(const std::string& filename, ImageData& image) {
    int width, height, channels;
    unsigned char* raw_data = stbi_load(filename.c_str(), &width, &height, &channels, 0);
    
    if (!raw_data) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        std::cerr << "STB Error: " << stbi_failure_reason() << std::endl;
        return false;
    }
    
    image.width = width;
    image.height = height;
    image.channels = channels;
    image.data.resize(width * height * channels);
    
    // Convert to float and normalize to [0, 1]
    for (int i = 0; i < width * height * channels; i++) {
        image.data[i] = static_cast<float>(raw_data[i]) / 255.0f;
    }
    
    stbi_image_free(raw_data);
    
    std::cout << "Loaded image: " << width << "x" << height << " with " << channels << " channels" << std::endl;
    return true;
}

bool save_image_jpeg(const std::string& filename, const ImageData& image, int quality) {
    std::vector<unsigned char> byte_data(image.width * image.height * image.channels);
    
    // Convert from float [0, 1] to byte [0, 255]
    for (size_t i = 0; i < byte_data.size(); i++) {
        float clamped = std::max(0.0f, std::min(1.0f, image.data[i]));
        byte_data[i] = static_cast<unsigned char>(clamped * 255.0f + 0.5f);
    }
    
    int result = stbi_write_jpg(filename.c_str(), image.width, image.height, image.channels, 
                                byte_data.data(), quality);
    
    if (result) {
        std::cout << "Saved image: " << filename << std::endl;
        return true;
    } else {
        std::cerr << "Failed to save image: " << filename << std::endl;
        return false;
    }
}

ImageData create_test_image(int width, int height) {
    ImageData image(width, height, 3);  // RGB
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            
            // Create a simple gradient pattern
            float r = static_cast<float>(x) / width;
            float g = static_cast<float>(y) / height;
            float b = 0.5f * (r + g);
            
            image.data[idx + 0] = r;      // Red
            image.data[idx + 1] = g;      // Green  
            image.data[idx + 2] = b;      // Blue
        }
    }
    
    std::cout << "Created test image: " << width << "x" << height << std::endl;
    return image;
}