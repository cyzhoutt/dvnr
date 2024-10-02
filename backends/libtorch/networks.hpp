#ifndef NETWORKS_HPP
#define NETWORKS_HPP


#pragma once

#include "pybind11/pybind11.h"
#include <torch/torch.h>
#include <cmath>
#include <vector>
#include <functional>
#include <fstream>
#include <map>
#include <stdio.h>

namespace py = pybind11;

class CppNetwork : public torch::nn::Module {
public:
    void to(torch::Device device);
    std::vector<torch::Tensor> parameters() const;
    std::map<std::string, torch::Tensor> named_parameters(bool recurse) const;
};

class MultiLayerPerceptron : public CppNetwork {
private:
    int n_output_dims;
    torch::nn::Linear first{nullptr};
    torch::nn::Linear last{nullptr};
    torch::nn::ModuleList hidden;
    std::function<torch::Tensor(torch::Tensor)> activation;
    std::function<torch::Tensor(torch::Tensor)> output_activation;

public:
    MultiLayerPerceptron(
        int64_t n_input_dims, int64_t n_output_dims, 
        int64_t n_hidden_layers, int64_t n_neurons, 
        std::string activation, std::string output_activation,
        bool bias = false
    );

    MultiLayerPerceptron(py::dict config, const py::dict& weights);

    torch::Tensor forward(torch::Tensor coords);
};

class HashGridEncoder : public CppNetwork {
private:
    int n_pos_dims;
    int n_levels;
    int n_features_per_level;
    int log2_hashmap_size;
    int base_resolution;
    float per_level_scale;
    int n_output_dims;
    std::vector<int64_t> embedding_offsets;
    std::vector<int64_t> embedding_lengths;
    torch::Tensor params;

public:
    HashGridEncoder(
        int n_pos_dims, 
        int n_levels, 
        int n_features_per_level,
        int log2_hashmap_size, 
        int base_resolution, 
        float per_level_scale
    );

    HashGridEncoder(py::dict config, const py::dict& weights);
    
    torch::Tensor forward(torch::Tensor coords);

private:
    static float grid_scale(int level, float per_level_scale, float base_resolution);

    static int grid_resolution(float scale);

    // Additional methods and static methods
    static torch::Tensor trilinear_interp_weights(const torch::Tensor& weights);

    // TODO should be able to do better in C++
    static std::pair<torch::Tensor, torch::Tensor> grid_indices(int scale, const torch::Tensor& coords);

    // TODO should be able to do better in C++
    static torch::Tensor hash_it(int hashmap_size, int resolution, const torch::Tensor& indices);

    // TODO should be able to do better in C++
    std::pair<torch::Tensor, torch::Tensor> access(const torch::Tensor& coords, int level);
};

// Something is missing in the pybind11 bindings for INR_Cpp
class INR_Cpp : public CppNetwork {
private:
    int n_enc_out;
    int n_enc_pad;
    int n_pad;
    std::shared_ptr<MultiLayerPerceptron> mlp;
    std::shared_ptr<HashGridEncoder> encoder;

public:
    INR_Cpp(
        int n_input_dims, int64_t n_output_dims, 
        int64_t n_hidden_layers, int64_t n_neurons, 
        int n_levels, 
        int n_features_per_level,
        int log2_hashmap_size, 
        int base_resolution, 
        float per_level_scale,
        std::string activation, 
        std::string output_activation
    );

    INR_Cpp(py::dict config, const py::dict& weights);
    
    torch::Tensor forward(torch::Tensor coords);
};
#endif