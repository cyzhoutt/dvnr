#include "backend.h"

#include <vnr/api.h>

#include <cuda/cuda_buffer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <iostream>
#include <fstream>
#include <cstring>

namespace dvnr {
namespace cpp_backend {

std::vector<uint8_t> CppVNR::get_network_params(bool fullprecision) const
{
  const nlohmann::json::binary_t& params = this->params["parameters"]["params_binary"];

  if (!fullprecision) {
    return params;
  }
  else {
    const __half* params_hp = (const __half*)params.data();
    const size_t n_params = params.size() / sizeof(__half);

    std::vector<uint8_t> output(n_params * sizeof(float));
    float* params_fp = (float*)output.data();

    for (int i = 0; i < n_params; ++i) {
      params_fp[i] = __half2float(params_hp[i]);
    }
    return output;
  }
}

void CppVNR::set_network_params(vnrVolume net, const std::vector<uint8_t>& params_data, bool fullprecision) 
{
  if (!fullprecision) {
    this->params["parameters"]["params_binary"] = params_data;
  }
  else {
    const float* params_fp = (const float*)params_data.data();
    const size_t n_params = params_data.size() / sizeof(float);

    nlohmann::json::binary_t output;
    output.resize(n_params * sizeof(__half));
    __half* params_hp = (__half*)output.data();

    for (int i = 0; i < n_params; ++i) {
      params_hp[i] = __float2half(params_fp[i]);
    }

    this->params["parameters"]["params_binary"] = std::move(output);
  }

  vnrNeuralVolumeSetParams(net, this->params);
}

}
}
