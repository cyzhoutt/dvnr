#include "networks.hpp"
#include <torch/extension.h>

#include <cmath>
#include <vector>
#include <functional>
#include <fstream>
#include <iostream>
#include <stdio.h>

namespace  F = torch::nn::functional;

constexpr int MLP_ALIGNMENT = 16;

auto coherent_prime_hash(torch::Tensor coords, int64_t log2_hashmap_size = 0) {
    std::vector<int64_t> primes = {1, 2654435761, 805459861, 3674653429,
                                   2097192037, 1434869437, 2165219737};

    auto xor_result = torch::zeros_like(coords).select(-1, 0);
    for (int64_t i = 0; i < coords.size(-1); ++i) {
        xor_result ^= coords.select(-1, i) * primes[i];
    }

    if (log2_hashmap_size == 0) {
        return xor_result;
    } else {
        return torch::tensor({(1 << log2_hashmap_size) - 1}, torch::dtype(torch::kInt64)).to(xor_result.device()) & xor_result;
    }
}

std::function<torch::Tensor(torch::Tensor)> find_activation(std::string activation) {
    if (activation == "None") {
        return [] (torch::Tensor x) { return x; };
    }
    else if (activation == "ReLU") {
        return [] (torch::Tensor x) { return F::relu(x, F::ReLUFuncOptions().inplace(true)); };
    }
    else {
        throw std::runtime_error("Activation function not found: " + activation);
    }
}

void init_weights(torch::nn::Module& network, py::dict weights) {
    torch::NoGradGuard no_grad;
    auto params = network.named_parameters(true);
    for (auto& el : weights) {
        auto key = el.first.cast<std::string>();
        auto val = el.second.cast<torch::Tensor>();
        if (params.contains(key)) {
            params[key].copy_(val);
        }
        else {
            std::cout << "Key not found: " << key << std::endl;
        }
    }
}


void 
CppNetwork::to(torch::Device device) { 
    Module::to(device); 
}

std::vector<torch::Tensor> 
CppNetwork::parameters() const {
    return Module::parameters(true);
}

std::map<std::string, torch::Tensor> 
CppNetwork::named_parameters(bool recurse) const {
    auto dict = Module::named_parameters(recurse);
    std::map<std::string, torch::Tensor> ret;
    for (auto& el : dict) {
        ret[el.key()] = el.value();
    }
    return ret;
}


MultiLayerPerceptron::MultiLayerPerceptron(
    int64_t n_input_dims, int64_t n_output_dims, 
    int64_t n_hidden_layers, int64_t n_neurons, 
    std::string activation, std::string output_activation,
    bool bias
)
    : n_output_dims(n_output_dims)
{
    // NOTE: "register_module" will automatically register the module IN ORDER
    //       To allow for the same behavior as the Python version, we need to 
    //       register the modules in the same order

    this->first = register_module("first", 
        torch::nn::Linear(torch::nn::LinearOptions(n_input_dims, n_neurons).bias(bias))
    );

    this->hidden = register_module("hidden", torch::nn::ModuleList());
    for (int64_t i = 0; i < n_hidden_layers - 1; ++i) {
        // NOTE: push_back will automatically register the module
        // https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/nn/modules/container/modulelist.h#L94
        this->hidden->push_back(
            torch::nn::Linear(torch::nn::LinearOptions(n_neurons, n_neurons).bias(bias))
        );
    }

    this->last = register_module("last", 
        torch::nn::Linear(torch::nn::LinearOptions(
            n_neurons,
            (n_output_dims + MLP_ALIGNMENT - 1) / MLP_ALIGNMENT * MLP_ALIGNMENT
        ).bias(bias))
    );

    this->activation = find_activation(activation);
    this->output_activation = find_activation(output_activation);
}

MultiLayerPerceptron::MultiLayerPerceptron(py::dict config, const py::dict& weights)
    : MultiLayerPerceptron(
        config["n_input_dims"].cast<int>(),
        config["n_output_dims"].cast<int>(),
        config["n_hidden_layers"].cast<int>(),
        config["n_neurons"].cast<int>(),
        config["activation"].cast<std::string>(),
        config["output_activation"].cast<std::string>(),
        false)
{
    init_weights(*this, weights);
}

torch::Tensor 
MultiLayerPerceptron::forward(torch::Tensor x) {
    x = activation(first->forward(x));
    for (size_t i = 0; i < hidden->size(); ++i) {
        x = activation(hidden->ptr<torch::nn::LinearImpl>(i)->forward(x));
    }
    return output_activation(last->forward(x)).index({ "...", torch::indexing::Slice(0, n_output_dims) });
}


HashGridEncoder::HashGridEncoder(
    int n_pos_dims, 
    int n_levels, 
    int n_features_per_level,
    int log2_hashmap_size, 
    int base_resolution, 
    float per_level_scale
)
    : n_pos_dims(n_pos_dims)
    , n_levels(n_levels)
    , n_features_per_level(n_features_per_level)
    , log2_hashmap_size(log2_hashmap_size)
    , base_resolution(base_resolution)
    , per_level_scale(per_level_scale)
    , n_output_dims(n_levels * n_features_per_level)
{
    TORCH_CHECK(n_pos_dims == 3, "Only 3 dimensional inputs are supported");
    std::vector<int64_t> embedding_offsets;
    std::vector<int64_t> embedding_lengths;
    int64_t offset = 0;
    for (int i = 0; i < n_levels; ++i) {
        float scale = grid_scale(i, per_level_scale, base_resolution);
        int resolution = grid_resolution(scale);
        int64_t length = std::pow(resolution, n_pos_dims);
        length = (length + 8 - 1) / 8 * 8;  // Align memory accesses
        length = std::min(length, static_cast<int64_t>(1 << log2_hashmap_size));
        embedding_offsets.push_back(offset);
        embedding_lengths.push_back(length);
        offset += length;
    }
    this->embedding_offsets = embedding_offsets;
    this->embedding_lengths = embedding_lengths;
    float scale = 1.0;
    this->params = register_parameter("params", torch::zeros(offset * n_features_per_level, torch::kFloat32));
    torch::nn::init::uniform_(params, -1e-4 * scale, 1e-4 * scale);
}

HashGridEncoder::HashGridEncoder(py::dict config, const py::dict& weights)
    : HashGridEncoder(
        config["n_pos_dims"].cast<int>(),
        config["n_levels"].cast<int>(),
        config["n_features_per_level"].cast<int>(),
        config["log2_hashmap_size"].cast<int>(),
        config["base_resolution"].cast<int>(),
        config["per_level_scale"].cast<double>()
    )
{
    init_weights(*this, weights);
}
    
torch::Tensor 
HashGridEncoder::forward(torch::Tensor coords) {
    coords = coords.contiguous().to(torch::kFloat32);
    torch::Tensor offsets_arr_tensor;
    torch::Tensor weights_arr_tensor;
    {
        torch::NoGradGuard no_grad;
        std::vector<torch::Tensor> weights_arr;
        std::vector<torch::Tensor> offsets_arr;
        for (int i = 0; i < n_levels; ++i) {
            auto pairAccess = access(coords, i);
            auto offsets = pairAccess.first;
            auto weights = pairAccess.second;
            offsets_arr.push_back(offsets.unsqueeze(1));
            weights_arr.push_back(weights.unsqueeze(1));
        }
        offsets_arr_tensor = torch::cat(offsets_arr, 1);
        weights_arr_tensor = torch::cat(weights_arr, 1);
        weights_arr_tensor = trilinear_interp_weights(weights_arr_tensor);
    }
    auto embeds_arr = torch::nn::functional::embedding(offsets_arr_tensor, params.reshape({-1, n_features_per_level}));
    auto out = (weights_arr_tensor.unsqueeze(-1) * embeds_arr).sum(-2);
    return out.reshape({-1, n_output_dims});
}

float 
HashGridEncoder::grid_scale(int level, float per_level_scale, float base_resolution) {
    // torch::NoGradGuard no_grad; // NOTE: this is not necessary because this function does not involve any Tensor operations
    return std::pow(2.0f, static_cast<float>(level) * std::log2(per_level_scale)) * base_resolution - 1.0f;
}

int 
HashGridEncoder::grid_resolution(float scale) {
    // torch::NoGradGuard no_grad; // NOTE: this is not necessary because this function does not involve any Tensor operations
    return static_cast<int>(std::ceil(scale)) + 1;
}

// Additional methods and static methods
torch::Tensor 
HashGridEncoder::trilinear_interp_weights(const torch::Tensor& weights) {
    torch::NoGradGuard no_grad;
    auto xs = weights.index({"...", 0});
    auto ys = weights.index({"...", 1});
    auto zs = weights.index({"...", 2});
    auto c0 = (1 - xs) * (1 - ys) * (1 - zs);
    auto c1 = (1 - xs) * (1 - ys) *      zs;
    auto c2 = (1 - xs) *      ys  * (1 - zs);
    auto c3 = (1 - xs) *      ys  *      zs;
    auto c4 =      xs  * (1 - ys) * (1 - zs);
    auto c5 =      xs  * (1 - ys) *      zs;
    auto c6 =      xs  *      ys  * (1 - zs);
    auto c7 =      xs  *      ys  *      zs;
    return torch::stack({c0, c1, c2, c3, c4, c5, c6, c7}, -1);
}

// TODO should be able to do better in C++
std::pair<torch::Tensor, torch::Tensor> 
HashGridEncoder::grid_indices(int scale, const torch::Tensor& coords) {
    torch::NoGradGuard no_grad;
    auto positions = (coords * scale + 0.5).to(torch::kFloat32);
    auto indices = torch::floor(positions).to(torch::kInt32); 
    positions = positions - indices;  // fractional part
    auto offsets = coords.new_empty({8, 3}, torch::kInt32).copy_(
        torch::tensor({{0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
                       {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1}}, torch::kInt32)); 
    indices = indices.unsqueeze(-2) + offsets.unsqueeze(0);
    return {indices, positions};
}


// TODO should be able to do better in C++
torch::Tensor 
HashGridEncoder::hash_it(int hashmap_size, int resolution, const torch::Tensor& indices) {
    torch::NoGradGuard no_grad;
    assert(indices.size(-1) == 3);
    uint32_t res = static_cast<uint32_t>(resolution);
    int stride = 1;
    auto output = torch::zeros_like(indices.index({"...", 0}));
    for (int dim = 0; dim < 3; ++dim) {
        output += indices.index({"...", dim}) * stride;
        stride *= res;  // Expecting integer overflow in scalar multiply
        if (stride > hashmap_size) break;
    }
    if (hashmap_size < stride) output =  coherent_prime_hash(indices);
    return output % hashmap_size;
}

// TODO should be able to do better in C++
std::pair<torch::Tensor, torch::Tensor> 
HashGridEncoder::access(const torch::Tensor& coords, int level) {
    torch::NoGradGuard no_grad;
    auto scale = grid_scale(level, per_level_scale, base_resolution);
    auto resolution = grid_resolution(scale);
    auto hashmap_size = embedding_lengths[level];
    auto [indices, fractions] = grid_indices(scale, coords);
    auto offsets = hash_it(hashmap_size, resolution, indices);
    return std::make_pair(offsets, fractions);
}


INR_Cpp::INR_Cpp(
    int n_input_dims, int64_t n_output_dims, 
    int64_t n_hidden_layers, int64_t n_neurons, 
    int n_levels, 
    int n_features_per_level,
    int log2_hashmap_size, 
    int base_resolution, 
    float per_level_scale,
    std::string activation, 
    std::string output_activation
)
    : n_enc_out(n_levels * n_features_per_level)
    , n_enc_pad((n_enc_out + MLP_ALIGNMENT - 1) / MLP_ALIGNMENT * MLP_ALIGNMENT)
    , n_pad(n_enc_pad - n_enc_out)
    , mlp(std::make_shared<MultiLayerPerceptron>(
        n_enc_pad, 
        n_output_dims, 
        n_hidden_layers, 
        n_neurons, 
        activation, 
        output_activation
    ))
    , encoder(std::make_shared<HashGridEncoder>(
        n_input_dims, 
        n_levels, 
        n_features_per_level,
        log2_hashmap_size, 
        base_resolution, 
        per_level_scale
    ))
{
    Module::register_module("mlp", mlp);
    Module::register_module("encoder", encoder);
}

INR_Cpp::INR_Cpp(py::dict config, const py::dict& weights)
    : INR_Cpp(
        config["n_input_dims"].cast<int>(),
        config["n_output_dims"].cast<int>(),
        config["n_hidden_layers"].cast<int>(),
        config["n_neurons"].cast<int>(),
        config["n_levels"].cast<int>(),
        config["n_features_per_level"].cast<int>(),
        config["log2_hashmap_size"].cast<int>(),
        config["base_resolution"].cast<int>(),
        config["per_level_scale"].cast<double>(),
        config["activation"].cast<std::string>(),
        config["output_activation"].cast<std::string>()
    )
{
    init_weights(*this, weights);
}
    
torch::Tensor INR_Cpp::forward(torch::Tensor coords) {
    auto h = encoder->forward(coords);
    h = F::pad(h, F::PadFuncOptions({ 0, n_pad }));
    return mlp->forward(h);
}


#ifdef EXTENSION_NAME
PYBIND11_MODULE(EXTENSION_NAME, m) {
    py::class_<CppNetwork, std::shared_ptr<CppNetwork>>(m, "CppNetwork")
        .def("to", &CppNetwork::to)
        .def("named_parameters", &CppNetwork::named_parameters)
        .def("parameters", &CppNetwork::parameters);
    py::class_<MultiLayerPerceptron, CppNetwork, std::shared_ptr<MultiLayerPerceptron>>(m, "MultiLayerPerceptron")        
        .def(py::init<py::dict, const py::dict&>())
        .def("forward", &MultiLayerPerceptron::forward);
    py::class_<HashGridEncoder, CppNetwork, std::shared_ptr<HashGridEncoder>>(m, "HashGridEncoder")        
        .def(py::init<py::dict, const py::dict&>())
        .def("forward", &HashGridEncoder::forward);
    py::class_<INR_Cpp, CppNetwork, std::shared_ptr<INR_Cpp>>(m, "INR_Cpp")
        .def(py::init<py::dict, const py::dict&>())
        .def("forward", &INR_Cpp::forward);
    m.doc() = "Generic Volume Data Sampler"; // optional module docstring
}
#endif
