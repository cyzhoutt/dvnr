#pragma once

#include <dvnr/dvnr_internal.h>

#include <json/json.hpp>

#include <string>
#include <memory>
#include <iostream>
#include <functional>
#include <stdexcept>

#ifdef NDEBUG
#define ASSERT_THROW(X, MSG) ((void)0)
#else
#define ASSERT_THROW(X, MSG) { if (!(X)) throw std::runtime_error(MSG); }
#endif

namespace dvnr {
namespace cpp_backend {

void initialize();
void finalize();
dvnr_t createVNR(const DataDesc& data, const ModelDesc& model, const OptimizerDesc& optimizer);
void decodeVNR(const dvnr_t& repr, const char * filename);
void decodeVNR(const dvnr_t& repr, float * data);

struct CppVNR : public DistributedVNR
{
    std::vector<uint8_t> get_network_params(bool fullprecision) const;
    void set_network_params(vnrVolume net, const std::vector<uint8_t>& params_data, bool fullprecision);
};

typedef std::shared_ptr<CppVNR> cpp_dvnr_t;

}
}
