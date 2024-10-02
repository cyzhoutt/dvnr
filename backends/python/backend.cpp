#include "backend.h"

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>

#include <iostream>
#include <fstream>
#include <cstring>

namespace dvnr::py_backend {

    // ------------------------------------------------------------------
    // Internal Data
    // ------------------------------------------------------------------

    bool interpreter_initialized = false;

    struct PyVNR : public DistributedVNR
    {
        std::shared_ptr<py::dict> states;
    };

    typedef std::shared_ptr<PyVNR> py_dvnr_t;

    template<typename T>
    void PyVNR_inner(const DataDesc& data, py_dvnr_t& repr)
    {
        if (!interpreter_initialized) throw std::runtime_error("python interpreter uninitialized");
        const auto func = py::module_::import("dvnr").attr("compress_ndarray");
        const auto stack = py::module_::import("numpy").attr("stack");
        const auto reshape = py::module_::import("numpy").attr("reshape");

        const ModelDesc& model = repr->model;
        const OptimizerDesc& optimizer = repr->optimizer;

        py::dict optional;
        if (!repr->minmax.is_empty()) {
            py::list minmax;
            minmax.append(repr->minmax.lower);
            minmax.append(repr->minmax.upper);
            optional["minmax"] = minmax;
        }
        optional["numfields"] = repr->numfields;

        // construct python zero-copy handler
        py::array_t<T> ndarray;
        if (data.fields_flatten) {
            py::list shape;
            shape.append((int64_t)repr->dims.long_product());
            shape.append((int64_t)repr->numfields);
            ndarray = reshape(
                py::array_t<T>(repr->dims.long_product() * repr->numfields, (T*)data.fields_flatten), 
                shape
            );
        }
        else {
            if (data.fields.size() != repr->numfields) throw std::runtime_error("mismatch number of fields");
            py::list array_list;
            for (int i = 0; i < data.fields.size(); ++i) {
                auto fdata = py::array_t<T>(repr->dims.long_product(), (T*)data.fields[i]);
                array_list.append(fdata);
            }
            ndarray = stack(array_list, "axis"_a=1);
        }

        // train by loading data in C++
        py::list ret = func(ndarray, 
            py::make_tuple(repr->dims.x, repr->dims.y, repr->dims.z), repr->dtype,
            **optional,
            **to_kwargs(model),
            **to_kwargs(optimizer),
            "evaluate"_a=true,
            "verbose"_a=optimizer.verbose
        );

        repr->mse  = py::float_(ret[1]);
        repr->psnr = py::float_(ret[2]);
        repr->time = py::float_(ret[3]);

        repr->states.reset(new py::dict(std::move(ret[0])));
    }

    void PyVNR_callback(const DataDesc& data, py_dvnr_t& repr)
    {
        if (!interpreter_initialized) throw std::runtime_error("python interpreter uninitialized");
        const auto func = py::module_::import("dvnr").attr("compress_callback");

        const ModelDesc& model = repr->model;
        const OptimizerDesc& optimizer = repr->optimizer;

        throw std::runtime_error("unimplemented");
    #if 0
        auto sampler = CallbackSampler(
            repr->numfields, 
            [=] (void* in, void* out, size_t len) -> void {
                data.callback(data.callback_context, in, out, len);
            }
        );

        py::list ret = func(sampler, 
            py::make_tuple(repr->dims.x, repr->dims.y, repr->dims.z), repr->dtype,
            **to_kwargs(model),
            **to_kwargs(optimizer),
            "evaluate"_a=true,
            "verbose"_a=optimizer.verbose
        );

        repr->mse  = py::float_(ret[1]);
        repr->psnr = py::float_(ret[2]);
        repr->time = py::float_(ret[3]);

        repr->states.reset(new py::dict(std::move(ret[0])));
    #endif
    }

    // ------------------------------------------------------------------
    // Serializer Module
    // ------------------------------------------------------------------

    #define safe_access(config, output, var) {                      \
        if (config.contains(#var)) {                                \
            output.var = config[#var].cast<decltype(output.var)>(); \
        }                                                           \
    }

    py::dict to_kwargs(const ModelDesc& desc)
    {
        return py::dict(
            "tcnn"_a=desc.tcnn,
            "n_hidden_layers"_a=desc.n_hidden_layers, 
            "n_neurons"_a=desc.n_neurons, 
            "n_levels"_a=desc.n_levels, 
            "n_features_per_level"_a=desc.n_features_per_level, 
            "base_resolution"_a=desc.base_resolution, 
            "log2_hashmap_size"_a=desc.log2_hashmap_size,
            "per_level_scale"_a=desc.per_level_scale
        );
    }

    template<> 
    ModelDesc from_kwargs<ModelDesc>(const py::dict& config)
    {
        ModelDesc model;
        safe_access(config, model, tcnn);
        safe_access(config, model, n_hidden_layers);
        safe_access(config, model, n_neurons);
        safe_access(config, model, n_levels);
        safe_access(config, model, n_features_per_level);
        safe_access(config, model, base_resolution);
        safe_access(config, model, log2_hashmap_size);
        safe_access(config, model, per_level_scale);
        return model;
    }

    // --------------------------------------------------------------------------------- // 

    py::dict to_kwargs(const OptimizerDesc& desc)
    {
        return py::dict(
            "max_steps"_a=desc.max_steps,
            // "psnr_target"_a=desc.psnr_target, // disable the use of PSNR target
            "lrate"_a=desc.lrate, 
            "lrate_decay"_a=desc.lrate_decay, 
            "batchsize"_a=desc.batchsize
        );
    }

    template<>
    OptimizerDesc from_kwargs<OptimizerDesc>(const py::dict& config)
    {
        OptimizerDesc ret;
        safe_access(config, ret, max_steps);
        // safe_access(config, ret, psnr_target);
        safe_access(config, ret, lrate);
        safe_access(config, ret, lrate_decay);
        safe_access(config, ret, batchsize);
        safe_access(config, ret, verbose);
        return ret;
    }

    // --------------------------------------------------------------------------------- // 

    #if 0
    py::dict to_kwargs(const MacroCell& desc)
    {
        throw std::runtime_error("TODO");
    }

    template<>
    dvnr::MacroCell from_kwargs<MacroCell>(const py::dict& config)
    {
        MacroCell ret;
        if (config.contains("volumedims")) {
            auto r = config["volumedims"].cast<std::array<int, 3>>();
            ret.volumedims.x = r[0];
            ret.volumedims.y = r[1];
            ret.volumedims.z = r[2];
        }
        if (config.contains("dims")) {
            auto r = config["dims"].cast<std::array<int, 3>>();
            ret.dims.x = r[0];
            ret.dims.y = r[1];
            ret.dims.z = r[2];
        }
        if (config.contains("spacings")) {
            auto r = config["spacings"].cast<std::array<float, 3>>();
            ret.spacings.x = r[0];
            ret.spacings.y = r[1];
            ret.spacings.z = r[2];
        }
        if (config.contains("value_ranges")) {
            auto r = config["value_ranges"]
                .cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            ASSERT_THROW(r.shape(2) == ret.dims.x, "wrong");
            ASSERT_THROW(r.shape(1) == ret.dims.y, "wrong");
            ASSERT_THROW(r.shape(0) == ret.dims.z, "wrong");
            ASSERT_THROW(r.shape(3) == 2, "wrong");
            ret.value_ranges.resize(ret.dims.long_product());
            std::memcpy(ret.value_ranges.data(), (vec2f*)r.data(), ret.dims.long_product() * sizeof(vec2f));
        }

        return ret;
    }
    #endif

    nlohmann::json serialize_model(const py::dict& states, const ModelDesc& model/*, const MacroCell& mc*/)
    {
        // using half = __half;
    
        size_t params_count;
        nlohmann::json::binary_t params_binary;
        {
            torch::Tensor tensor = states["model.params"].cast<torch::Tensor>().to(torch::kHalf).to(torch::kCPU);
            params_count = tensor.size(0);
            params_binary.resize(params_count * sizeof(torch::Half));
            // CUDA_CHECK(cudaMemcpy(params_binary.data(), tensor.data_ptr(), params_count * sizeof(half), cudaMemcpyDeviceToHost));
            std::memcpy(params_binary.data(), tensor.data_ptr(), params_count * sizeof(torch::Half));
        }

        // nlohmann::json::binary_t mcdata;
        // {
        //     mcdata.resize(mc.value_ranges.size() * sizeof(vec2f));
        //     std::memcpy(mcdata.data(), mc.value_ranges.data(), mcdata.size());
        // }

        nlohmann::json root;

        root["volume"] = {
            { "dims", {
                // { "x", mc.volumedims.x },
                // { "y", mc.volumedims.y },
                // { "z", mc.volumedims.z }
            }}
        };

        // // Currently, macrocell only applies to scalar field volume
        // if (!mc.empty()) {
        //     root["macrocell"] = {
        //         { "groundtruth", false },
        //         { "dims", {
        //                 { "x", mc.dims.x },
        //                 { "y", mc.dims.y },
        //                 { "z", mc.dims.z },
        //             }
        //         },
        //         { "spacings", {
        //                 { "x", mc.spacings.x },
        //                 { "y", mc.spacings.y },
        //                 { "z", mc.spacings.z },
        //             }
        //         },
        //         { "data", mcdata },
        //     };
        // }
        
        root["parameters"] = {
            { "n_params", params_count },
            { "params_binary", std::move(params_binary) },
        };

        root["model"] = {
            { "loss", { { "otype", "L1" } } },
            { "encoding",  
                {
                    { "otype", "HashGrid" },
                    { "n_levels", model.n_levels },
                    { "n_features_per_level", model.n_features_per_level },
                    { "log2_hashmap_size", model.log2_hashmap_size },
                    { "base_resolution", model.base_resolution },
                    { "per_level_scale", model.per_level_scale },
                } 
            },
            { "network",
                {
                    { "otype", "FullyFusedMLP" },
                    { "n_neurons", model.n_neurons },
                    { "n_hidden_layers", model.n_hidden_layers },
                    { "activation", "ReLU" },
                    { "output_activation", "None" },
                } 
            }
        };

        return std::move(root);
    }

}

    // ------------------------------------------------------------------
    // 
    // ------------------------------------------------------------------
    using namespace dvnr;
    using namespace dvnr::py_backend;

    void dvnr::py_backend::initialize()
    {
        py::initialize_interpreter();
        {
            // modify system search path
            auto syspath = py::list(py::module_::import("sys").attr("path"));
            syspath.append(DVNR_ROOT_DIR);
            syspath.append(DVNR_BUILD_DIR);
            py::module_::import("dvnr_ext");
        }
        interpreter_initialized = true;

        py::print("py::initialize", "dvnr", "sep"_a="-"); // print banner
    }

    void dvnr::py_backend::finalize()
    {
        py::print("py::finalize", "dvnr", "sep"_a="-"); // print banner

        py::finalize_interpreter();
        interpreter_initialized = false;
    }

    dvnr_t 
    dvnr::py_backend::createVNR(const DataDesc& data, const ModelDesc& model, const OptimizerDesc& optimizer)
    {
        py_dvnr_t repr = std::make_shared<PyVNR>();

        repr->set_data_desc(data);

        repr->model = model;
        repr->optimizer = optimizer;

        // TODO support multi-variable fields
        // MacroCell mc;
        // if (repr->numfields == 1) { 
        //     mc.compute(data);
        // }

        // we choose callback sampler if available
        if (data.callback) {
            PyVNR_callback(data, repr);
        }
        else {
            const auto dtype = std::string(data.dtype);
            if      (dtype == "uint8")   PyVNR_inner<uint8_t >(data, repr);
            else if (dtype == "uint16")  PyVNR_inner<uint16_t>(data, repr);
            else if (dtype == "uint32")  PyVNR_inner<uint32_t>(data, repr);
            else if (dtype == "uint64")  PyVNR_inner<uint64_t>(data, repr);
            else if (dtype == "int8")    PyVNR_inner<int8_t  >(data, repr);
            else if (dtype == "int16")   PyVNR_inner<int16_t >(data, repr); 
            else if (dtype == "int32")   PyVNR_inner<int32_t >(data, repr);
            else if (dtype == "int64")   PyVNR_inner<int64_t >(data, repr);
            else if (dtype == "float32") PyVNR_inner<float   >(data, repr);
            else if (dtype == "float64") PyVNR_inner<double  >(data, repr);
            else throw std::runtime_error("unknown data type: " + dtype);
        }

        repr->params = serialize_model(*repr->states, model/*, mc*/);

        const nlohmann::json::binary_t& binary = repr->params["parameters"]["params_binary"];
        repr->compressor.compressed = binary;

        repr->uncompressed_mse = repr->mse;

        return repr;
    }

    void 
    dvnr::py_backend::decodeVNR(const dvnr_t& _repr, const char* filename)
    {
        const PyVNR* repr = (const PyVNR*)_repr.get();

        if (!interpreter_initialized) throw std::runtime_error("python interpreter uninitialized");
        const auto decoder = py::module_::import("dvnr").attr("decode_to_file");

        const ModelDesc& model = repr->model;
        const OptimizerDesc& optimizer = repr->optimizer;

        py::dict optional;
        optional["numfields"] = repr->numfields;
        if (!repr->minmax.is_empty()) {
            py::list minmax;
            minmax.append(repr->minmax.lower);
            minmax.append(repr->minmax.upper);
            optional["minmax"] = minmax;
        }

        decoder(filename, *repr->states, 
            py::make_tuple(repr->dims.x, repr->dims.y, repr->dims.z), repr->dtype,
            **optional,
            **to_kwargs(model),
            **to_kwargs(optimizer)
        );
    }

    void 
    dvnr::py_backend::decodeVNR(const dvnr_t& _repr, float* data)
    {
        const PyVNR* repr = (const PyVNR*)_repr.get();

        if (!interpreter_initialized) throw std::runtime_error("python interpreter uninitialized");
        const auto decoder = py::module_::import("dvnr").attr("decode_to_ndarray");

        const ModelDesc& model = repr->model;
        const OptimizerDesc& optimizer = repr->optimizer;

        py::dict optional;
        optional["numfields"] = repr->numfields;
        if (!repr->minmax.is_empty()) {
            py::list minmax;
            minmax.append(repr->minmax.lower);
            minmax.append(repr->minmax.upper);
            optional["minmax"] = minmax;
        }

        auto ndarray = py::memoryview::from_buffer(
            data, // buffer pointer
            { repr->dims.long_product() * repr->numfields }, // shape (rows, cols)
            { sizeof(float) } // strides in bytes
        );

        decoder(ndarray, *repr->states, 
            py::make_tuple(repr->dims.x, repr->dims.y, repr->dims.z), repr->dtype,
            **optional,
            **to_kwargs(model),
            **to_kwargs(optimizer)
        );
    }

