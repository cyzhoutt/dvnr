#pragma once

#include <dvnr/dvnr_internal.h>

#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include <json/json.hpp>

#include <string>
#include <memory>
#include <iostream>
#include <functional>
#include <stdexcept>

namespace py = pybind11;
using namespace pybind11::literals; // to bring in the `_a` literal

#ifdef NDEBUG
#define ASSERT_THROW(X, MSG) ((void)0)
#else
#define ASSERT_THROW(X, MSG) { if (!(X)) throw std::runtime_error(MSG); }
#endif

namespace dvnr {
namespace py_backend {

void initialize();
void finalize();
dvnr_t createVNR(const DataDesc& data, const ModelDesc& model, const OptimizerDesc& optimizer);
void decodeVNR(const dvnr_t& repr, const char * filename);
void decodeVNR(const dvnr_t& repr, float * data);

// ------------------------------------------------------------------
// Serializer Module
// ------------------------------------------------------------------
template<typename T>
T from_kwargs(const py::dict& config);

py::dict to_kwargs(const ModelDesc& desc);
template<> ModelDesc from_kwargs<ModelDesc>(const py::dict& config);

py::dict to_kwargs(const OptimizerDesc& desc);
template<> OptimizerDesc from_kwargs<OptimizerDesc>(const py::dict& config);

py::dict to_kwargs(const MacroCell& desc);
template<> MacroCell from_kwargs<MacroCell>(const py::dict& config);

nlohmann::json serialize_model(const py::dict& states, const ModelDesc& model/*, const MacroCell& mc*/);

// ------------------------------------------------------------------
// Sampler Module
// ------------------------------------------------------------------
class CallbackSampler {
public:
    int numfields;
    std::function<void(void*,void*,size_t)> callback;

    std::string msg = "callback sampler constructed in C++"; // debug message

public:
    CallbackSampler(int numfields, std::function<void(void*,void*,size_t)> callback) 
        : numfields(numfields)
        , callback(callback) 
    {}

    torch::Tensor sample(const torch::Tensor& object) const; // defined in binding.cpp

    std::vector<double> get_value_range() const {
        return { 0.0, 1.0 };
    }

    std::vector<double> get_value_range_unnormalized() const {
        throw std::runtime_error("unimplemented");
    }

    void print() { std::cout << msg << std::endl; }
};

}
}
