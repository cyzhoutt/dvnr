#pragma once

// do not use std::string in the API due to c++ ABI compatibility

#include "config.h"
#include "dvnr.h"
#include "instantvnr/vnr/api.h"
#include <string>

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace vnr {
    
inline int dtype_size(std::string dtype)
{
    if      (dtype == "uint8")   return sizeof(uint8_t);
    else if (dtype == "uint16")  return sizeof(uint16_t);
    else if (dtype == "uint32")  return sizeof(uint32_t);
    else if (dtype == "uint64")  return sizeof(uint64_t);
    else if (dtype == "int8")    return sizeof(int8_t);
    else if (dtype == "int16")   return sizeof(int16_t);
    else if (dtype == "int32")   return sizeof(int32_t);
    else if (dtype == "int64")   return sizeof(int64_t);
    else if (dtype == "float32") return sizeof(float);
    else if (dtype == "float64") return sizeof(double);
    throw std::runtime_error("unknown data type: " + dtype);
}

inline ValueType dtype_type(std::string dtype)
{
    if      (dtype == "uint8")   return VALUE_TYPE_UINT8;
    else if (dtype == "uint16")  return VALUE_TYPE_UINT16;
    else if (dtype == "uint32")  return VALUE_TYPE_UINT32;
    else if (dtype == "uint64")  return VALUE_TYPE_UINT64;
    else if (dtype == "int8")    return VALUE_TYPE_INT8;
    else if (dtype == "int16")   return VALUE_TYPE_INT16;
    else if (dtype == "int32")   return VALUE_TYPE_INT32;
    else if (dtype == "int64")   return VALUE_TYPE_INT64;
    else if (dtype == "float32") return VALUE_TYPE_FLOAT;
    else if (dtype == "float64") return VALUE_TYPE_DOUBLE;
    throw std::runtime_error("unknown data type: " + dtype);
}

}

// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace dvnr {

// --------------------------------------------------------------------------------- // 

constexpr int N_POS_DIMS = 3;

struct CompressedParameters 
{
    size_t n1, n2;

    // input parameters
    double accuracy_mlp = 0.0; // non-positive accuracy should disable weight compression
    double accuracy_enc = 0.0; // non-positive accuracy should disable weight compression

    // cached parameters
    ModelDesc model;
    size_t n_params = 0;
    std::vector<size_t> offset_table;
    std::vector<size_t> offset_table_comp;

    // compression result
    std::vector<uint8_t> compressed;

    // statistics
    double compression_ratio = 1.0;
    double compression_time  = 0.0;
    double rmse_prev = 0.0;
    double ssim_prev = 0.0;

    double compress(const float* params, size_t n_params, const ModelDesc&); // return compression time
    std::vector<uint8_t> decompress() const;
};

struct DistributedVNR
{
    virtual ~DistributedVNR() = default;
    virtual void dummy() {}
    int numfields = 1;

    vnr::vec3i dims = -1;
    std::string dtype;

    bool enable_clipping = false;
    vnr::box3f clipbox;

    bool enable_scaling = false;
    vnr::vec3f scaling;

    vnr::range1f minmax;

    ModelDesc model;
    OptimizerDesc optimizer;

    nlohmann::json params;
    CompressedParameters compressor;

    size_t total_steps = 0;
    double mse;
    double psnr;
    double ssim;
    double time;
    size_t vram = 0;

    double zfp_ratio = 1.0;
    double zfp_time = 0.0;
    double uncompressed_mse = 0.0;
    double uncompressed_ssim = 0.0;

    void set_data_desc(const DataDesc& data) 
    {
        this->dims.x = data.dimx;
        this->dims.y = data.dimy;
        this->dims.z = data.dimz;
        this->dtype  = data.dtype;

        this->numfields = data.numfields;

        this->minmax.lower = data.min;
        this->minmax.upper = data.max;

        this->enable_clipping = data.enable_clipping;
        this->clipbox.lower.x = data.clipbox[0];
        this->clipbox.lower.y = data.clipbox[1];
        this->clipbox.lower.z = data.clipbox[2];
        this->clipbox.upper.x = data.clipbox[3];
        this->clipbox.upper.y = data.clipbox[4];
        this->clipbox.upper.z = data.clipbox[5];

        this->enable_scaling = data.enable_scaling;
        this->scaling.x = data.scaling[0];
        this->scaling.y = data.scaling[1];
        this->scaling.z = data.scaling[2];
    }

    DataDesc get_data_desc() const
    {
        DataDesc data;
        data.dimx = this->dims.x;
        data.dimy = this->dims.y;
        data.dimz = this->dims.z;
        data.dtype = this->dtype.c_str();

        data.numfields = this->numfields;

        data.min = this->minmax.lower;
        data.max = this->minmax.upper;

        data.enable_clipping = this->enable_clipping;
        data.clipbox[0] = this->clipbox.lower.x;
        data.clipbox[1] = this->clipbox.lower.y;
        data.clipbox[2] = this->clipbox.lower.z;
        data.clipbox[3] = this->clipbox.upper.x;
        data.clipbox[4] = this->clipbox.upper.y;
        data.clipbox[5] = this->clipbox.upper.z;

        data.enable_scaling = this->enable_scaling;
        data.scaling[0] = this->scaling.x;
        data.scaling[1] = this->scaling.y;
        data.scaling[2] = this->scaling.z;

        return data;
    }

    double compression_ratio() const 
    {
        const auto& params_data = dvnrGetParams(this);
        const size_t numbytes_model = params_data.size();
        const size_t numbytes_data = this->dims.long_product() * this->numfields * vnr::dtype_size(this->dtype);
        return double(numbytes_data) / double(numbytes_model);
    }
    // ~DistributedVNR() {}
};

namespace math = gdt;
using vec2f = math::vec2f;
using vec2i = math::vec2i;
using vec3f = math::vec3f;
using vec3i = math::vec3i;
using vec4f = math::vec4f;
using vec4i = math::vec4i;
using range1i = math::range1i;
using range1f = math::range1f;
using math::max;
using math::min;


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

namespace cpp_backend
{
    void initialize();
    void finalize();
    dvnr_t createVNR(const DataDesc& data, const ModelDesc& model, const OptimizerDesc& optimizer);
    void decodeVNR(const dvnr_t& repr, const char * filename);
    void decodeVNR(const dvnr_t& repr, float * data);
}

namespace py_backend
{
    void initialize();
    void finalize();
    dvnr_t createVNR(const DataDesc& data, const ModelDesc& model, const OptimizerDesc& optimizer);
    void decodeVNR(const dvnr_t& repr, const char * filename);
    void decodeVNR(const dvnr_t& repr, float * data);
}
namespace lib_backend
{
    void initialize();
    void finalize();
    dvnr_t createVNR(const DataDesc& data, const ModelDesc& model, const OptimizerDesc& optimizer);
    void decodeVNR(const dvnr_t& repr, const char * filename);
    void decodeVNR(const dvnr_t& repr, float * data);
}


// ------------------------------------------------------------------
//
// ------------------------------------------------------------------

struct MacroCell 
{
    vnr::math::vec3i volumedims{};
    vnr::math::vec3i dims{};
    vnr::math::vec3f spacings{};
    std::vector<vnr::math::vec2f> value_ranges;
    void compute(const DataDesc& data);
    void compute_explicit(const DataDesc& data);
    void compute_direct  (const DataDesc& data);
    bool empty() const { return dims.long_product() == 0; }
};

// ------------------------------------------------------------------
// Various Helper Functions
// ------------------------------------------------------------------
void saveJPG(const std::string &fname, vnr::vec2i size, const vnr::vec4f* pixels);

}
