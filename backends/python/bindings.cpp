#include "backend.h"

#include <torch/extension.h>

#include <iostream>

using dvnr::py_backend::CallbackSampler;

static int n_output_dims() { return 1; }

static void* void_data_ptr(torch::Tensor& tensor) {
	switch (tensor.scalar_type()) {
		case torch::kFloat32: return tensor.data_ptr<float>();
		case torch::kHalf: return tensor.data_ptr<torch::Half>();
		default: throw std::runtime_error{"Unknown precision torch->void"};
	}
}

torch::Tensor CallbackSampler::sample(const torch::Tensor& input) const
{
    const uint32_t batch_size = input.size(0);
    const uint32_t numfields = 1;
    torch::Tensor output = torch::empty({ batch_size, numfields }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    float* in = input.data_ptr<float>();
    void* out = void_data_ptr(output);

    callback(in, out, batch_size);

    return output;
}

PYBIND11_MODULE(dvnr_ext, m) {
    m.doc() = "Generic Volume Data Sampler"; // optional module docstring
    py::class_<CallbackSampler>(m, "GenericSampler")
        .def("print", &CallbackSampler::print)
        .def("sample",  &CallbackSampler::sample)
        .def("get_value_range", &CallbackSampler::get_value_range)
        .def("get_value_range_unnormalized", &CallbackSampler::get_value_range_unnormalized);
}
