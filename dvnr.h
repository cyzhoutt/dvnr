#pragma once

// do not use std::string in the API due to c++ ABI compatibility

#include <memory>
#include <vector>
#include <limits>

namespace dvnr {

struct DataDesc {
    int dimx = -1;
    int dimy = -1;
    int dimz = -1;
    const char * dtype;

    int numfields = 1;

    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();

    bool enable_clipping = false;
    float clipbox[6] = { 0 };

    bool enable_scaling = false;
    float scaling[3] = { 1, 1, 1 };

    // variant 1 (flattened ND array)
    std::vector<void*> fields;
    void* fields_flatten = nullptr;

    // variant 2
    void* callback_context = nullptr;
    void (*callback)(void* ctx, void*, void*, unsigned long long) = nullptr;

    const char* fieldname = nullptr;
};

struct ModelDesc {
    bool tcnn=false;
    int n_hidden_layers=4;
    int n_neurons=64;
    int n_levels=8;
    int n_features_per_level=4;
    int base_resolution=4;
    int log2_hashmap_size=19;
    float per_level_scale=2.f;
    // zfp compression
    float zfp_accuracy_mlp=2e-2f;
    float zfp_accuracy_enc=4e-2f;
};

struct OptimizerDesc {
    float lrate=1e-2f;
    int lrate_decay=250;
    int batchsize=32*64;
    bool verbose=false;
    // termination criteria
    int max_steps=50;
    float target_loss=-1;
};

struct SceneDesc {
    struct {
        float from[3];
        float at[3];
        float up[3];
    } camera;

    // when `tfn_file` is NULL, check `tfn_data` will be used. 
    const char * tfn_file { nullptr };
    struct {
        float* colors;
        float* alphas;
        int resolution = 0;
        float range[2] = { 0.f, 1.f };
    } tfn_data;

    int fbsize[2];
    int spp=1;

    enum {
        RAY_MARCHING=5,
        RAY_MARCHING_GRADIENT=8,
        RAY_MARCHING_SINGLE_SHOT=11,
        PATH_TRACING=14,
    } mode = RAY_MARCHING;
    bool enable_denoising = false;
    float volume_sampling_rate = 1.f;

    // when `output_filename` is NULL, check `output_rgba` will be used. 
    // one of them must be set.
    const char  * output_filename { nullptr };
    const float * output_rgba { nullptr };
};

struct TriangleMeshDesc {
    std::vector<size_t> numvertices;
    std::vector<float*> vertices; // of size 3 * num_vertices
    float extraction_time = 0.f;
    size_t peak_vram = 0;
};

struct DistributedVNR;

}

typedef std::shared_ptr<dvnr::DistributedVNR> dvnr_t;
typedef dvnr::DataDesc dvnr_data_t;
typedef dvnr::ModelDesc dvnr_model_t;
typedef dvnr::OptimizerDesc dvnr_optim_t;
typedef dvnr::SceneDesc dvnr_scene_t;
typedef dvnr::TriangleMeshDesc dvnr_trimesh_t;

void dvnrInitialize(const char* backend);
void dvnrFinalize();

dvnr_t dvnrCreate(const dvnr_data_t& data, const dvnr_model_t& model, const dvnr_optim_t& optimizer);
void dvnrDecode(const dvnr_t& repr, const char * filename);
void dvnrDecode(const dvnr_t& repr, float * data);

void dvnrSerialize(const dvnr_t& repr, const char * filename);
void dvnrRender(const dvnr_t& repr, const dvnr_scene_t& scene);
void dvnrFree(dvnr_t& repr);

dvnr_trimesh_t dvnrExtractMesh(const dvnr_t& repr, std::vector<float> isovalues);
dvnr_trimesh_t dvnrExtractMesh(const dvnr_data_t& data, std::vector<float> isovalues);

double dvnrGetRMSE(const dvnr_t& repr);
double dvnrGetPSNR(const dvnr_t& repr);
double dvnrGetSSIM(const dvnr_t& repr);
double dvnrGetCompressTime(const dvnr_t& repr);
double dvnrGetCompressionRatio(const dvnr_t& repr);
size_t dvnrGetVRAM(const dvnr_t& repr);
size_t dvnrGetTotalSteps(const dvnr_t& repr);
double dvnrGetUncompressedRMSE(const dvnr_t& repr);
double dvnrGetUncompressedSSIM(const dvnr_t& repr);
double dvnrGetWeightCompressionRatio(const dvnr_t& repr);
double dvnrGetWeightCompressionTime(const dvnr_t& repr);

const std::vector<uint8_t>& dvnrGetParams(const dvnr::DistributedVNR* repr);
const std::vector<uint8_t>& dvnrGetParams(const dvnr_t& repr);

// helper functions //

namespace dvnr {

struct VolumeDesc_Structured {
  DataDesc shape;
  const char * filename;
  unsigned long long offset;
  bool is_big_endian;

  void* dst;
};

struct VolumeDesc_Unstructured { 
    /* TODO integrate umesh */ 
};

}

void dvnrLoadData(dvnr::VolumeDesc_Structured& desc);
void dvnrLoadData(dvnr::VolumeDesc_Unstructured& desc);

void dvnrGPUMemory_ResetPeakUsage();
size_t dvnrGPUMemory_PeakUsage();
void dvnrGPUMemory_PrintUsage(const char* str);
void dvnrGPUMemory_FreeTemporary();
