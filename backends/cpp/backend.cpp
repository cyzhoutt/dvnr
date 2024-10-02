#include "backend.h"

#include <iostream>
#include <fstream>
#include <cstring>

#include <vidi_progress_bar.h>
#include <vidi_highperformance_timer.h>

namespace dvnr::cpp_backend {

using Timer = vidi::details::HighPerformanceTimer;

void to_json(const ModelDesc& desc, vnrJson& model) {
    model["encoding"] = {
        { "otype", "HashGrid" },
        { "n_levels", desc.n_levels },
        { "n_features_per_level", desc.n_features_per_level },
        { "log2_hashmap_size", desc.log2_hashmap_size },
        { "base_resolution", desc.base_resolution },
        { "per_level_scale", desc.per_level_scale },
    };
    model["network"] = {
        { "otype", "FullyFusedMLP" },
        { "activation", "ReLU" },
        { "output_activation", "Exponential" },
        { "n_neurons", desc.n_neurons },
        { "n_hidden_layers", desc.n_hidden_layers },
    };
}

void to_json(const OptimizerDesc& desc, vnrJson& optim) {
    vnrJson adam = {
        { "otype", "Adam" },
        { "learning_rate", desc.lrate },
        { "beta1", 0.9 },
        { "beta2", 0.999 },
        { "epsilon", 1e-09 },
        // { "epsilon", 1e-12 },
        { "l2_reg", 1e-09 }
    };
    optim["loss"] = {
        // { "otype", "L2" },
        { "otype", "L1" },
    };
    optim["optimizer"] = {
        { "otype", "ExponentialDecay" },
        { "decay_start", desc.lrate_decay },
        { "decay_interval", desc.lrate_decay },
        { "decay_end", 1e20 /* never ends */ },
        { "decay_base", 0.8 },
        { "nested", adam },
    };
}

}

// ------------------------------------------------------------------
// 
// ------------------------------------------------------------------
using namespace dvnr;
using namespace dvnr::cpp_backend;

void dvnr::cpp_backend::initialize()
{
    // no-op
}

void dvnr::cpp_backend::finalize()
{
    // no-op
}

void 
train_target_step(vnrVolume net, int target_step, bool verbose)
{
    vnrNeuralVolumeTrain(net, target_step, true, verbose);
}

void 
train_target_loss(vnrVolume net, float target_loss, bool verbose)
{
    while (true) {
        vnrNeuralVolumeTrain(net, 512, true, verbose);
        float loss = vnrNeuralVolumeGetTrainingLoss(net);
        if (loss < target_loss) break;
        if (vnrNeuralVolumeGetTrainingStep(net) >= 20480*4) { break; }
    }
}

double 
train_network(vnrVolume net, const cpp_dvnr_t& repr, const std::string& key, vnrJson model)
{
    static vnrJson cache;

    // load from the parameter cache
    vnrJson &init = cache[key];
    bool invalid = init.is_null();
    if (!invalid) {
        bool disabled = getenv("DVNR_DISABLE_WEIGHT_CACHING");
        bool consistent = (init["model"]["encoding"] == model["encoding"]) && (init["model"]["network"] == model["network"]);
        if (!disabled && consistent) {
            vnrNeuralVolumeSetParams(net, init);
        }
    }

    Timer timer; 
    timer.start();
    if (repr->optimizer.target_loss > 0) {
        train_target_loss(net, repr->optimizer.target_loss, repr->optimizer.verbose);
    }
    else {
        train_target_step(net, repr->optimizer.max_steps, repr->optimizer.verbose);
    }
    timer.stop();

    // fetch model weights
    vnrNeuralVolumeSerializeParams(net, init);
    repr->params = init;
    repr->uncompressed_mse = vnrNeuralVolumeGetMSE(net, false);
    repr->uncompressed_ssim = vnrNeuralVolumeGetSSIM(net, false);
    repr->total_steps = vnrNeuralVolumeGetTrainingStep(net);
    return timer.milliseconds() / 1000.0;
}

double 
weight_compression(vnrVolume net, const cpp_dvnr_t& repr)
{
    std::vector<uint8_t> params_fp = repr->get_network_params(true);

    size_t n_params = params_fp.size() / sizeof(float);
    float* params = (float*)params_fp.data();

    repr->compressor.accuracy_enc = repr->model.zfp_accuracy_enc;
    repr->compressor.accuracy_mlp = repr->model.zfp_accuracy_mlp;

    size_t n_mlp_params = vnrNeuralVolumeGetNBytesMultilayerPerceptron(net) / 2;
    size_t n_enc_params = vnrNeuralVolumeGetNBytesEncoding(net) / 2;
    float* params_mlp = params;
    float* params_enc = params + n_mlp_params;
    if (n_mlp_params + n_enc_params != n_params) {
        throw std::runtime_error("incorrect number of parameters");
    }
    repr->compressor.n1 = n_enc_params;
    repr->compressor.n2 = n_mlp_params;

    double time_comp = repr->compressor.compress(params, n_params, repr->model);
    params_fp = std::move(repr->compressor.decompress());
    repr->set_network_params(net, params_fp, true);

    repr->zfp_ratio = (float)(n_params * /*sizeof(half)=*/2) / repr->compressor.compressed.size();

    return time_comp;
}

dvnr_t 
dvnr::cpp_backend::createVNR(const DataDesc& data, const ModelDesc& _model, const OptimizerDesc& _optimizer)
{
    cpp_dvnr_t repr = std::make_shared<CppVNR>();

    repr->set_data_desc(data);
    repr->model = _model;
    repr->optimizer = _optimizer;

    vnrVolume simple_volume;
    vnrVolume neural_volume;

    if (repr->numfields != 1) {
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
        simple_volume = vnrCreateSimpleVolume(dptr, repr->dims, repr->dtype, repr->minmax, "GPU");
    }

    vnrJson model; 
    to_json(repr->model, model); 
    to_json(repr->optimizer, model);

    neural_volume = vnrCreateNeuralVolume(model, simple_volume, false, repr->optimizer.batchsize);

    // train the model
    double training_time = train_network(neural_volume, repr, data.fieldname, model);

    // weight compression
    double compression_time = 0.0;
    if (!getenv("DVNR_DISABLE_WEIGHT_COMPRESSION")) {
        compression_time = weight_compression(neural_volume, repr);
    }

    // analysis performance
    const float rangesize = 1.0; // NOTE we fix data range == 1 to match python implementation
    float mse  = vnrNeuralVolumeGetMSE(neural_volume, repr->optimizer.verbose);
    repr->mse  = mse;
    repr->psnr = (float)(10. * log10(rangesize * rangesize / mse));
    repr->ssim = vnrNeuralVolumeGetSSIM(neural_volume, repr->optimizer.verbose);
    repr->time = training_time + compression_time;
    repr->zfp_time = compression_time;

    size_t vram_peak = 0;
    size_t vram_curr = 0;
    vnrMemoryQuery(NULL, NULL, &vram_peak, &vram_curr);
    repr->vram = vram_peak;

    return repr;
}

void 
dvnr::cpp_backend::decodeVNR(const dvnr_t& repr, const char* filename)
{
    vnrVolume volume = vnrCreateNeuralVolume(
        repr->params["model"], repr->dims
    );
    vnrNeuralVolumeSetParams(volume, repr->params);
    vnrNeuralVolumeDecodeInference(volume, filename);
}

void 
dvnr::cpp_backend::decodeVNR(const dvnr_t& repr, float* data)
{
    vnrVolume volume = vnrCreateNeuralVolume(
        repr->params["model"], repr->dims
    );
    vnrNeuralVolumeSetParams(volume, repr->params);
    vnrNeuralVolumeDecode(volume, data);
}
