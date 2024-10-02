#include "dvnr_internal.h"

#include <iostream>
#include <fstream>
#include <cstring>

using namespace dvnr;

namespace dvnr {
static std::string backend = "invalid";
}

// --------------------------------------------------------------------------------- // 

void dvnrFree(dvnr_t& repr) { repr.reset(); } 
double dvnrGetRMSE(const dvnr_t& repr) { return std::sqrt(repr->mse); }
double dvnrGetPSNR(const dvnr_t& repr) { return repr->psnr; }
double dvnrGetSSIM(const dvnr_t& repr) { return repr->ssim; }
double dvnrGetCompressTime(const dvnr_t& repr) { return repr->time; }
double dvnrGetCompressionRatio(const dvnr_t& repr) { return repr->compression_ratio(); }
double dvnrGetUncompressedRMSE(const dvnr_t& repr) { return std::sqrt(repr->uncompressed_mse); }
double dvnrGetUncompressedSSIM(const dvnr_t& repr) { return repr->uncompressed_ssim; }
double dvnrGetWeightCompressionRatio(const dvnr_t& repr) { return repr->zfp_ratio; }
double dvnrGetWeightCompressionTime(const dvnr_t& repr)  { return repr->zfp_time;  }

size_t dvnrGetVRAM(const dvnr_t& repr) { return repr->vram; }
size_t dvnrGetTotalSteps(const dvnr_t& repr) { return repr->total_steps; }

const std::vector<uint8_t>& dvnrGetParams(const dvnr_t& repr) { return dvnrGetParams(repr.get()); }
const std::vector<uint8_t>& dvnrGetParams(const dvnr::DistributedVNR* repr) {
    if (repr->compressor.compressed.empty()) {
        throw std::runtime_error("compressed parameters not available");
    }
    return repr->compressor.compressed;
}

void dvnrSerialize(const dvnr_t& repr, const char* filename)
{
    const auto broot = nlohmann::json::to_bson(repr->params);
    std::ofstream ofs(filename, std::ios::binary | std::ios::out);
    ofs.write((char*)broot.data(), broot.size());
    ofs.close();
}

#if 0
void dvnrRender(const dvnr_t& repr, const SceneDesc& scene)
{
    using namespace vnr;

    vnrCamera camera;
    vnrVolume volume;
    vnrTransferFunction tfn;
    vnrRenderer renderer;

    vec3f cam_from(scene.camera.from[0], scene.camera.from[1], scene.camera.from[2]);
    vec3f cam_at(scene.camera.at[0], scene.camera.at[1], scene.camera.at[2]);
    vec3f cam_up(scene.camera.up[0], scene.camera.up[1], scene.camera.up[2]);

    camera = vnrCreateCamera();
    vnrCameraSet(camera, cam_from, cam_at, cam_up);

    // 1179622.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov (okay)

    volume = vnrCreateNeuralVolume(
        repr->params["model"], repr->dims
    );

    if (repr->enable_clipping) {
        vnrVolumeSetClippingBox(volume, repr->clipbox.lower, repr->clipbox.upper);
    }

    if (repr->enable_scaling) {
        vnrVolumeSetScaling(volume, repr->scaling);
    }

    vnrNeuralVolumeSetParams(volume, repr->params);

    // 1179623.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov (OKAY (1.890148 - 1.889748) GB = 400 KB)

    if (scene.tfn_file && !std::string(scene.tfn_file).empty()) {
        tfn = vnrCreateTransferFunction(scene.tfn_file);
    }
    else {
        if (scene.tfn_data.resolution == 0) throw std::runtime_error("empty transfer function");

        tfn = vnrCreateTransferFunction();

        std::vector<vnr::vec3f> colors(scene.tfn_data.resolution);
        std::vector<vnr::vec2f> alphas(scene.tfn_data.resolution);
        for (int i = 0; i < scene.tfn_data.resolution; ++i) {
            colors[i] = vnr::vec3f(scene.tfn_data.colors[3*i], scene.tfn_data.colors[3*i+1], scene.tfn_data.colors[3*i+2]);
            alphas[i] = vnr::vec2f((float)i/scene.tfn_data.resolution, scene.tfn_data.alphas[i]);
        }
        vnrTransferFunctionSetColor(tfn, colors);
        vnrTransferFunctionSetAlpha(tfn, alphas);
        vnrTransferFunctionSetValueRange(tfn, vnr::range1f(scene.tfn_data.range[0], scene.tfn_data.range[1]));
    }

    // 1179625.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov (OKAY (1.892117 -1.886234)GB = 5.883 MB)

    renderer = vnrCreateRenderer(volume);

    // 1179633.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov (BAD (2.157349 - 1.865547)GB = 291.802 MB)
    vnrRendererSetTransferFunction(renderer, tfn);
    vnrRendererSetCamera(renderer, camera);
    
    vnrRendererSetMode(renderer, (int)scene.mode);
    vnrRendererSetDenoiser(renderer, scene.enable_denoising);
    vnrRendererSetVolumeSamplingRate(renderer, scene.volume_sampling_rate);

    vnrRendererSetFramebufferSize(renderer, vec2i(scene.fbsize[0], scene.fbsize[1]));

    for (int i = 0; i < scene.spp; ++i) vnrRender(renderer);

    const vec4f *pixels = vnrRendererMapFrame(renderer);

    if (scene.output_filename) {
        saveJPG(scene.output_filename, vec2i(scene.fbsize[0], scene.fbsize[1]), pixels);
    }
    else if (scene.output_rgba) {
        std::copy(pixels, pixels + scene.fbsize[0] * scene.fbsize[1], (vec4f*)scene.output_rgba);
    }
    else {
        throw std::runtime_error("one of 'output_filename' and 'output_rgba' must be set.");
    }

    camera.reset();
    volume.reset();
    tfn.reset();
    renderer.reset();
}
#endif

#if 0
dvnr_trimesh_t dvnrExtractMesh(const vnrVolume& volume, std::vector<float>& isovalues)
{
    // extract mesh
    dvnr_trimesh_t output;
    output.numvertices.resize(isovalues.size());
    output.vertices.resize(isovalues.size());

    std::vector<vnrIsosurface> isosurfaces;
    for (int i = 0; i < isovalues.size(); ++i) {
        isosurfaces.emplace_back(vnrIsosurface{ isovalues[i], (vnr::vec3f**)&output.vertices[i], &output.numvertices[i], 0.0 });
    }

    vnrMarchingCube(volume, isosurfaces, /*output_to_cuda_memory*/ false);

    output.extraction_time = 0.f;
    for (int i = 0; i < isovalues.size(); ++i) {
        output.extraction_time += (float)isosurfaces[i].et;
    }

    size_t vram_peak = 0;
    vnrMemoryQuery(NULL, NULL, &vram_peak, NULL);
    output.peak_vram = vram_peak;

    return output;
}

dvnr_trimesh_t dvnrExtractMesh(const dvnr_t& repr, std::vector<float> isovalues)
{
    vnrVolume volume = vnrCreateNeuralVolume(
        repr->params["model"], repr->dims
    );
    vnrNeuralVolumeSetParams(volume, repr->params);
    return dvnrExtractMesh(volume, isovalues);
}

dvnr_trimesh_t dvnrExtractMesh(const dvnr_data_t& data, std::vector<float> isovalues)
{
    vnrVolume simple_volume;
    if (data.numfields != 1) {
        throw std::runtime_error("multivariate data not supported");
    }
    // we choose callback sampler if available
    if (data.callback) {
        throw std::runtime_error("callback sampler not implemented");
    }
    else {
        void* dptr;
        if (data.fields_flatten) {
            dptr = data.fields_flatten;
        }
        else {
            if (data.fields.size() != data.numfields) throw std::runtime_error("mismatch number of fields");
            dptr = data.fields[0];
        }
        auto dims = vnr::vec3i(data.dimx, data.dimy, data.dimz);
        auto dtype = std::string(data.dtype);
        auto minmax = vnr::range1f(data.min, data.max);
        simple_volume = vnrCreateSimpleVolume(dptr, dims, dtype, minmax, "GPU");
    }

    return dvnrExtractMesh(simple_volume, isovalues);
}
#endif

// --------------------------------------------------------------------------------- // 

dvnr_t dvnrCreate(const DataDesc& data, const ModelDesc& model, const OptimizerDesc& optimizer) {
#ifdef ENABLE_DVNR_PYTHON_BACKEND
    if (backend == "python") return py_backend::createVNR(data, model, optimizer);
#endif
#ifdef ENABLE_DVNR_CPP_BACKEND
    if (backend == "cpp") return cpp_backend::createVNR(data, model, optimizer);
#endif
#ifdef ENABLE_DVNR_LIB_BACKEND
    if (backend == "lib") return lib_backend::createVNR(data, model, optimizer);
#endif
    throw std::runtime_error("invalid backend: " + backend);
}

void dvnrDecode(const dvnr_t& repr, const char * filename) {
#ifdef ENABLE_DVNR_PYTHON_BACKEND
    if (backend == "python") return py_backend::decodeVNR(repr, filename);
#endif
#ifdef ENABLE_DVNR_CPP_BACKEND
    if (backend == "cpp") return cpp_backend::decodeVNR(repr, filename);
#endif
#ifdef ENABLE_DVNR_LIB_BACKEND
    if (backend == "lib") return lib_backend::decodeVNR(repr, filename);
#endif
    throw std::runtime_error("invalid backend: " + backend);
}

void dvnrDecode(const dvnr_t& repr, float * data) {
#ifdef ENABLE_DVNR_PYTHON_BACKEND
    if (backend == "python") return py_backend::decodeVNR(repr, data);
#endif
#ifdef ENABLE_DVNR_CPP_BACKEND
    if (backend == "cpp") return cpp_backend::decodeVNR(repr, data);
#endif
#ifdef ENABLE_DVNR_LIB_BACKEND
    if (backend == "lib") return lib_backend::decodeVNR(repr, data);
#endif
    throw std::runtime_error("invalid backend: " + backend);
}

void dvnrInitialize(const char* _backend) {
    backend = _backend;
    // vnrCompilationStatus("[dvnr]");
#ifdef ENABLE_DVNR_PYTHON_BACKEND
    if (backend == "python") return py_backend::initialize();
#endif
#ifdef ENABLE_DVNR_CPP_BACKEND
    if (backend == "cpp") return cpp_backend::initialize();
#endif
#ifdef ENABLE_DVNR_LIB_BACKEND
    if (backend == "lib") return lib_backend::initialize();
#endif
    throw std::runtime_error("invalid backend: " + backend);
}

void dvnrFinalize() {
#ifdef ENABLE_DVNR_PYTHON_BACKEND
    if (backend == "python") return py_backend::finalize();
#endif
#ifdef ENABLE_DVNR_CPP_BACKEND
    if (backend == "cpp") return cpp_backend::finalize();
#endif
#ifdef ENABLE_DVNR_LIB_BACKEND
    if (backend == "lib") return lib_backend::finalize();
#endif
    throw std::runtime_error("invalid backend: " + backend);
}

// --------------------------------------------------------------------------------- // 
