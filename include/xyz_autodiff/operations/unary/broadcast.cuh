#pragma once

#include <xyz_autodiff/concept/variable.cuh>
#include <xyz_autodiff/concept/operation_node.cuh>

namespace xyz_autodiff::op {

// BroadcastOperator: Efficiently broadcasts a length-1 Variable to length N
// No intermediate storage needed - just references the source value
template <typename Input, std::size_t OutputSize>
requires DifferentiableVariableConcept<Input> && (Input::size == 1) && (OutputSize > 1)
class BroadcastOperator {
public:
    using value_type = typename Input::value_type;
    static constexpr std::size_t size = OutputSize;

private:
    Input& input_;
    mutable std::uint8_t ref_count_ = 0;  // DAG対応: 参照カウント

public:
    // Constructor takes reference to input Variable
    __device__ BroadcastOperator(Input& input) 
        : input_(input) {}

    // DifferentiableVariableConcept requirements

    // Index access - returns the broadcasted value for any index
    __device__ const value_type& operator[](std::size_t i) const {
        return input_[0];  // All indices return the same value
    }
    
    __device__ value_type& operator[](std::size_t i) {
        return const_cast<value_type&>(input_[0]);  // Non-const version for compatibility
    }

    // Gradient access - virtual gradient (accumulated from all positions)
    __device__ const value_type& grad(std::size_t i) const {
        return input_.grad(0);
    }

    // Add gradient - accumulate for later propagation to input
    __device__ void add_grad(std::size_t i, value_type grad_value) {
        input_.add_grad(0, grad_value);
    }

    // Zero gradient
    __device__ void zero_grad() {
        input_.zero_grad();
    }

    // OperationNode concept requirements

    // Forward pass - no computation needed, values are accessed on-demand
    __device__ void forward() {
        if constexpr (OperationNode<Input>) {
            input_.forward();
            input_.increment_ref_count();
        }
    }

    // Backward pass - propagate accumulated gradients to input
    __device__ void backward() {
        if constexpr (OperationNode<Input>) {
            if (input_.decrement_ref_count_and_check()) {
                input_.backward();
            }
        }
    }

    // Numerical backward pass
    __device__ void backward_numerical(value_type delta = value_type(1e-5)) {
        // 入力がOperationNodeの場合、参照カウントを減らしてから条件付きbackward_numerical
        if constexpr (OperationNode<Input>) {
            if (input_.decrement_ref_count_and_check()) {
                input_.backward_numerical(delta);
            }
        }
    }

    // Additional utility methods

    // Run both forward and backward
    __device__ void run() {
        forward();
        backward();
    }

    // Run with numerical gradients
    __device__ void run_numerical(value_type delta = value_type(1e-5)) {
        forward();
        backward_numerical(delta);
    }

    __device__ void increment_ref_count() const {
        ref_count_++;
    }

    __device__ bool decrement_ref_count_and_check() const {
        return --ref_count_ == 0;
    }
};

// Factory function for broadcast operation
template <std::size_t OutputSize, typename Input>
requires DifferentiableVariableConcept<Input> && (Input::size == 1) && (OutputSize > 1)
__device__ auto broadcast(Input& input) {
    return BroadcastOperator<Input, OutputSize>(input);
}

} // namespace xyz_autodiff::op