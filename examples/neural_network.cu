#include <xyz_autodiff/xyz_autodiff.h>
#include <iostream>
#include <vector>
#include <random>

class SimpleNN {
public:
    SimpleNN(int input_size, int hidden_size, int output_size) 
        : w1_({input_size, hidden_size}, true)
        , b1_({1, hidden_size}, true)
        , w2_({hidden_size, output_size}, true)
        , b2_({1, output_size}, true) {
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        std::vector<float> w1_data(input_size * hidden_size);
        std::vector<float> b1_data(hidden_size, 0.0f);
        std::vector<float> w2_data(hidden_size * output_size);
        std::vector<float> b2_data(output_size, 0.0f);
        
        for (auto& val : w1_data) val = dist(gen);
        for (auto& val : w2_data) val = dist(gen);
        
        w1_.from_cpu(w1_data.data());
        b1_.from_cpu(b1_data.data());
        w2_.from_cpu(w2_data.data());
        b2_.from_cpu(b2_data.data());
    }
    
    xyz_autodiff::Tensor forward(const xyz_autodiff::Tensor& input) {
        auto h1 = input.matmul(w1_) + b1_;
        auto a1 = h1.relu();
        auto h2 = a1.matmul(w2_) + b2_;
        return h2.sigmoid();
    }
    
    void zero_grad() {
        w1_.zero_grad();
        b1_.zero_grad();
        w2_.zero_grad();
        b2_.zero_grad();
    }
    
private:
    xyz_autodiff::Tensor w1_, b1_, w2_, b2_;
};

int main() {
    xyz_autodiff::initialize();
    
    SimpleNN network(4, 8, 1);
    
    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> target_data = {1.0f};
    
    xyz_autodiff::Tensor input({1, 4}, false);
    xyz_autodiff::Tensor target({1, 1}, false);
    
    input.from_cpu(input_data.data());
    target.from_cpu(target_data.data());
    
    std::cout << "Neural Network Training Example:\n";
    
    for (int epoch = 0; epoch < 10; ++epoch) {
        network.zero_grad();
        
        auto output = network.forward(input);
        auto diff = output - target;
        auto loss = (diff * diff).mean();
        
        loss.backward();
        
        std::vector<float> loss_val(1);
        loss.to_cpu(loss_val.data());
        
        std::cout << "Epoch " << epoch << ", Loss: " << loss_val[0] << std::endl;
    }
    
    xyz_autodiff::cleanup();
    return 0;
}