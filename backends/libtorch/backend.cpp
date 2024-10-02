#include "backend.h"
#include "networks.hpp"
#include <json/json.hpp>
#include "sampler/sampler.h"
#include <iostream>
#include <string>
#include <chrono>
#include <stdio.h>
#include <map>
#include <torch/torch.h>
#include <vidi_progress_bar.h>
#include <vidi_highperformance_timer.h>
#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/stl.h> // for converting STL containers
#include <unistd.h>
#include <fcntl.h>
#include <dvnr_internal.h>
#include <torch/extension.h> 


namespace py = pybind11;

namespace dvnr::lib_backend {
   
    struct LibVNR : public DistributedVNR
    {
        // py::dict weights;
        std::map<std::string, torch::Tensor> weights;
    };

    typedef std::shared_ptr<LibVNR> lib_dvnr_t;


using namespace dvnr;
using INR_Base = INR_Cpp;
using namespace dvnr::lib_backend;

void init_weights(torch::nn::Module& network, std::map<std::string, torch::Tensor> weights) {
    torch::NoGradGuard no_grad;
    auto params = network.named_parameters(true);
    for (auto& el : weights) {
        auto key = el.first;
        auto val = el.second;
        if (params.contains(key)) {
            params[key].copy_(val);
        }
        else {
            std::cout << "Key not found: " << key << std::endl;
        }
    }
}
torch::Tensor generate_grid(int dimx, int dimy, int dimz, torch::Device device) {
    torch::Tensor x = torch::linspace(0, dimx - 1, dimx, torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor y = torch::linspace(0, dimy - 1, dimy, torch::TensorOptions().dtype(torch::kInt32).device(device));
    torch::Tensor z = torch::linspace(0, dimz - 1, dimz, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto grid = torch::meshgrid({ x, y, z }, "ij");
    auto coordinates = torch::stack({ grid[0], grid[1], grid[2] }, 
        3 // should be 3 rather than -1 here, because grid[0], grid[1], grid[2] only have 3 dimensions, not 4
    ).permute({2, 1, 0, 3});
    return coordinates.reshape({ -1, 3 }).to(torch::kFloat);
}


McReturn compute_macrocell(dvnr::vec3i& vdims, INR_Base& model, int mcsize_mip) {
    auto device = model.parameters().front().device(); // get the device of the model

    const int mcsize = 1 << mcsize_mip;
    const dvnr::vec3i mcdims = (vdims + mcsize - 1) / mcsize;

    auto fopt = torch::TensorOptions().device(device).dtype(torch::kFloat32);

    auto vscale  = torch::tensor({ 1.f / vdims[0],  1.f / vdims[1],  1.f / vdims[2] }, fopt);
    auto mcrange = torch::zeros({ mcdims[2], mcdims[1], mcdims[0], 2 }, fopt);

    auto grid = generate_grid(mcsize + 1, mcsize + 1, mcsize + 1, device) - 0.5;

    auto n = grid.size(0);
    auto m = mcdims[1] * mcdims[0];

    auto offset = torch::zeros({ 3 }, fopt);
    for (int64_t iz = 0; iz < mcdims[2]; ++iz) {
        offset[2] = iz;   
        auto o = generate_grid(mcdims[0], mcdims[1], 1, device) + offset;
        auto x = grid.repeat({m, 1}).reshape({m, n, 3}) + o.unsqueeze(1) * (float)mcsize;
        auto y = model.forward((x.reshape({-1, 3}) * vscale).clamp(0.f, 1.f)).reshape({m, n, -1});
        auto vmin = std::get<0>(y.min(1));
        auto vmax = std::get<0>(y.max(1));

        // for (int64_t k1 = 0; k1 < mcdims[1]; ++k1) {
        // std::cout << "- ";
        // for (int64_t k0 = 0; k0 < mcdims[0]; ++k0) {
        //     std::cout << std::fixed << std::setprecision(6) << vmax[k1*mcdims[0] + k0].item();
        //     std::cout << " ";
        // }
        // std::cout << std::endl;
        // }
        // std::cout << std::endl;

        mcrange[iz].select(-1, 0) = vmin.reshape({ mcdims[1], mcdims[0] });
        mcrange[iz].select(-1, 1) = vmax.reshape({ mcdims[1], mcdims[0] });
    }

    // remap the value range
    mcrange.select(-1, 0) -= 1;
    mcrange.select(-1, 1) += 1;
    mcrange = mcrange.to(torch::kCPU).contiguous();

    McReturn ret;
    ret.volumedims = vdims;
    ret.dims = mcdims;
    ret.spacings = (float)mcsize / (dvnr::vec3f)vdims;
    ret.value_ranges = std::vector<float>(mcrange.data_ptr<float>(), mcrange.data_ptr<float>() + mcrange.numel());
    return ret;
}

void save_json_binary(const nlohmann::json& root, std::string filename)
{
  const auto broot = nlohmann::json::to_bson(root);
  std::ofstream ofs(filename, std::ios::binary | std::ios::out);
  ofs.write((char*)broot.data(), broot.size());
  ofs.close();
}



nlohmann::json create_inr_scene(const std::string& fn, dvnr::vec3i& dims, INR_Base& model, nlohmann::json& encoding_config, nlohmann::json& network_config)
{
    torch::NoGradGuard no_grad;

    McReturn mc = compute_macrocell(dims, model);
	nlohmann::json::binary_t mcdata;
    mcdata.resize(mc.value_ranges.size() * sizeof(float));
    std::memcpy(mcdata.data(), mc.value_ranges.data(), mcdata.size());

    std::vector<torch::Tensor> params;
    for(auto& param : model.parameters()){
        params.push_back(param.flatten());
    }
    torch::Tensor params_halfs = torch::cat(params,0).contiguous().to(at::kHalf).to(torch::kCPU);
    assert(params_halfs.dim() == 1);
    size_t ncount = params_halfs.size(0);
    size_t nbytes = ncount * sizeof(torch::Half);
    std::cout << "params_halfs: " << ncount << std::endl;

	nlohmann::json::binary_t params_binary;
	params_binary.resize(nbytes);
    std::memcpy(params_binary.data(), params_halfs.data_ptr<torch::Half>(), params_binary.size());
    std::cout << "params_binary: " << params_binary.size() << " Bytes"<< std::endl;

    nlohmann::json root;
    root["macrocell"] = nlohmann::json{
        { "data", mcdata },
        { "dims", {
            { "x", mc.dims[0] }, 
            { "y", mc.dims[1] }, 
            { "z", mc.dims[2] }
        } },
        { "groundtruth", false },
        { "spacings", {
            { "x", mc.spacings[0] }, 
            { "y", mc.spacings[1] }, 
            { "z", mc.spacings[2] }
        } }
    };
    root["model"] = nlohmann::json{
        { "encoding", encoding_config },
        { "network",  network_config  },
        { "loss", {{ "otype", "L1" }} }
    };
    root["parameters"] = {
        { "n_params", ncount },
        { "params_binary", std::move(params_binary) }
    };
    root["volume"] = nlohmann::json{
        { "dims", {
            { "x", dims[0] }, 
            { "y", dims[1] }, 
            { "z", dims[2] }
        }}
    };

    save_json_binary(root, fn);
    return root;
}

   
// typedef std::shared_ptr<LibVNR> lib_dvnr_t;
void initialize()
{
    py::initialize_interpreter();

}

void finalize()
{
    py::finalize_interpreter();

}

dvnr_t createVNR(const DataDesc& desc, const ModelDesc& modelParam, const OptimizerDesc& descoptim){
    lib_dvnr_t repr = std::make_shared<LibVNR>();

    repr->set_data_desc(desc);

    repr->model = modelParam;
    repr->optimizer = descoptim;        
            
            torch::Device device = torch::cuda::is_available() 
            ? torch::Device(torch::kCUDA) 
    #if defined(ENABLE_XPU)
            : torch::Device(torch::kXPU);
    #elif defined(ENABLE_MPS)
            : torch::Device(torch::kMPS);
    #else   
            : torch::Device(torch::kCPU);
    #endif
        if (device == torch::kCPU) {
            std::cout << "Using CPU !!!" << std::endl;
        }

    dvnr::VolumeDataStructured ddata;
    // // desc.filename = "/Users/adazhou/Desktop/distributed-vnr/data/engine.raw"; // change here !!
    // // desc.filename = "/home/qadwu/Work/dvnr/data/engine_256x256x128_uint8.raw"; // change here !!

    // fdata.filename = "/workspace/data/engine_256x256x128_uint8.raw"; // change here !!
    ddata.dims = dvnr::vec3i(desc.dimx, desc.dimy, desc.dimz);
    ddata.type = dvnr::UINT8;
    ddata.spacing = dvnr::vec3f(1, 1, 1);
    ddata.n_channels = 1;
    ddata.data = (const uint8_t *)desc.fields_flatten;

    auto sampler = dvnr::create_sampler("openvkl", ddata);
    auto dims = dvnr::vec3i(desc.dimx, desc.dimy, desc.dimz);

    // std::cout << "@ " << __FILE__ << ":" << __LINE__ << std::endl;

    nlohmann::json network_configs;
    network_configs["otype"] = "FullyFusedMLP";
    network_configs["activation"] = "ReLU";
    network_configs["output_activation"] = "ReLU";
    network_configs["n_neurons"] = modelParam.n_neurons;    
    network_configs["n_hidden_layers"] = modelParam.n_hidden_layers; 
    // network_configs["feedback_alignment"] = false; // not needed anymore

    // std::cout << "@ " << __FILE__ << ":" << __LINE__ << std::endl;

    nlohmann::json encoding_configs;
    encoding_configs["otype"] = "HashGrid";
    encoding_configs["n_levels"] = modelParam.n_levels;
    encoding_configs["n_features_per_level"] = modelParam.n_features_per_level; 
    encoding_configs["log2_hashmap_size"] = modelParam.log2_hashmap_size;  
    encoding_configs["base_resolution"] = modelParam.base_resolution;  
    encoding_configs["per_level_scale"] = modelParam.per_level_scale;

    // std::cout << "@ " << __FILE__ << ":" << __LINE__ << std::endl;

    // Create an instance of INR_Base with the given configurations
    // Something is missing in the pybind11 bindings for INR_Cpp
    INR_Cpp imodel(3, 1, 
        network_configs["n_hidden_layers"], 
        network_configs["n_neurons"], 
        encoding_configs["n_levels"], 
        encoding_configs["n_features_per_level"], 
        encoding_configs["log2_hashmap_size"],
        encoding_configs["base_resolution"], 
        encoding_configs["per_level_scale"], 
        network_configs["activation"], 
        network_configs["output_activation"]
    );

    // std::cout << "@ " << __FILE__ << ":" << __LINE__ << std::endl;

    imodel.to(device);

    // Print model parameters
    int64_t numvoxels = desc.dimx * desc.dimy * desc.dimz;
    int64_t batchsize = descoptim.batchsize;
    int epochs = 1;
    int64_t multiplier = (numvoxels + batchsize - 1) / batchsize;

    torch::optim::Adam optimizer(imodel.parameters(), torch::optim::AdamOptions(descoptim.lrate)
        .betas(std::make_tuple(0.9, 0.999))
        .eps(1e-08)
        .weight_decay(1e-15)
        .amsgrad(false));

    // Set up the learning rate scheduler
    torch::optim::StepLR scheduler(optimizer, 16 * multiplier, 0.8);

    // Define L1 loss function
    torch::nn::L1Loss loss_function;
    auto loss_fn = [&] (const torch::Tensor& outputs, const torch::Tensor& targets) -> torch::Tensor {
        // Check that shapes of outputs and targets are the same
        assert(outputs.sizes() == targets.sizes());
        // Compute and return the loss
        return loss_function(outputs, targets);
    };

    torch::Tensor coords  = torch::empty({batchsize, 3}, torch::kFloat32).to(device);
    torch::Tensor targets = torch::empty({batchsize, 1}, torch::kFloat32).to(device);

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < multiplier*epochs+1; i++) {
        optimizer.zero_grad();
        {
            coords  = coords.to(torch::kCPU);
            targets = targets.to(torch::kCPU);
            sampler->sample(coords.data_ptr<float>(), targets.data_ptr<float>(), dvnr::vec3f(0), dvnr::vec3f(1), batchsize);
            coords  = coords.to(device);
            targets = targets.to(device);
            // std::cout << "targets: " << targets.max().item() << " " << targets.min().item() << std::endl;
        }
        auto values = imodel.forward(coords);
        auto loss = loss_fn(values.squeeze(), targets.squeeze()); // Assuming MSE loss for example

        loss.backward();
        optimizer.step();
        scheduler.step();

        if (i % 100 == 0) {
            std::cout << "Epoch: " << i << " Loss: " << loss.item<float>() << std::endl;
        }
    }

    // End timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Elapsed time: " << elapsed.count() << " s\n";

    // repr->params = imodel.parameters();
    // repr->mse = loss;


    repr->params = create_inr_scene("model.json", dims, imodel, encoding_configs, network_configs);
    std::cout <<"xdimension is" <<repr->params["macrocell"]["dims"]["x"] << std::endl;
    auto named_params = imodel.named_parameters(true);
    repr->weights.clear();  // Clear any existing data in weights
    for (const auto& p : named_params) {
        std::cout << "Tensor name: " << p.first << ", Tensor size: " << p.second.sizes() << std::endl;
        repr->weights[p.first] = p.second;
    };
    
    return repr;


}


void decodeVNR(const dvnr_t& repr, float* data)
{
    
    std::cout<<"entering decodeVNR 0"<<std::endl;
    const LibVNR* repr_casted = (const LibVNR*)(repr.get());
    // const auto& repr_ref = repr.get();  // Now repr_ref is an lvalue
    // LibVNR& repr_casted = const_cast<LibVNR&>(dynamic_cast<const LibVNR&>(repr_ref));
    std::cout<<"entering decodeVNR 1"<<std::endl;
    INR_Base model(3, 1, 
        repr_casted->params["model"]["network"]["n_hidden_layers"], 
        repr_casted->params["model"]["network"]["n_neurons"], 
        repr_casted->params["model"]["encoding"]["n_levels"], 
        repr_casted->params["model"]["encoding"]["n_features_per_level"], 
        repr_casted->params["model"]["encoding"]["log2_hashmap_size"],
        repr_casted->params["model"]["encoding"]["base_resolution"], 
        repr_casted->params["model"]["encoding"]["per_level_scale"], 
        repr_casted->params["model"]["network"]["activation"], 
        repr_casted->params["model"]["network"]["output_activation"]
    );
    init_weights(model, repr_casted->weights);
  
    std::cout<<"entering decodeVNR 2"<<std::endl;
    
    std::cout<<"might here!"<<std::endl;

    // auto input_to_grid = py::module_::import("dvnr").attr("input_to_grid");

    // dvnr::vec3i dims = dvnr::vec3i(repr_casted->params["volume"]["dims"]["x"],repr_casted->params["volume"]["dims"]["y"],repr_casted->params["volume"]["dims"]["z"]);
    
    // py::object matrix = input_to_grid(dims);
    
    // torch::Tensor matr = matrix.cast<torch::Tensor>();
    
    torch::Tensor result;
    // model.zero_grad();
    // result = model.forward(matr);
    int dims[3];
    dims[0] = repr_casted->params["macrocell"]["dims"]["x"];
    dims[0]+=1;
    dims[1] = repr_casted->params["macrocell"]["dims"]["y"];
    dims[1]+=1;
    dims[2] = repr_casted->params["macrocell"]["dims"]["z"];
    dims[2]+=1;
    // std::cout <<"xdimension is" <<repr_casted->params["macrocell"]["dims"]["x"] << std::endl;
    // double coords = new double[dims[0] * dims[1] * dims[2]*3];
    std::vector<double> coords;
    // coords.clear();
    // memset(coords, 0, sizeof(coords));
    // coords.reserve(3*dims[0] * dims[1] * dims[2]);
    // int i = 0;
    model.zero_grad();
    std::cout<<"before loop"<<std::endl;
    for(int x=0; x< dims[0]; x++){
        for(int y=0; y<dims[1];y++){
            for(int z=0;z<dims[2];z++){
                // std::vector<double> temp;
                double j = x/(dims[0]);
                coords.push_back(j);
                double k = y/(dims[1]);
                coords.push_back(k);
                double l = z/(dims[2]);
                coords.push_back(l);
            }
        }
    }
    std::cout<< "before blob" <<std::endl;
    auto tcoords = torch::tensor(coords);
    tcoords.view({dims[0]*dims[1]*dims[2], 3});
    std::cout<< "before forward" << tcoords.sizes() <<std::endl;
    result = model.forward(tcoords);
    result = result.reshape({dims[2],dims[1],dims[0]});

    int64_t num_elements = result.numel();

    int64_t element_size = result.element_size();

    int64_t total_size_in_bytes = num_elements * element_size;

    
    std::memcpy(data, result.data_ptr(), total_size_in_bytes);
    // free(repr_casted);

    return;
}

void decodeVNR(const dvnr_t& repr, const char* filename){
    std::cout<<"entering decodeVNR 0"<<std::endl;
    const LibVNR* repr_casted = (const LibVNR*)(repr.get());
    // const auto& repr_ref = repr.get();  // Now repr_ref is an lvalue
    // LibVNR& repr_casted = const_cast<LibVNR&>(dynamic_cast<const LibVNR&>(repr_ref));
    std::cout<<"entering decodeVNR 1"<<std::endl;
    INR_Base model(3, 1, 
        repr_casted->params["model"]["network"]["n_hidden_layers"], 
        repr_casted->params["model"]["network"]["n_neurons"], 
        repr_casted->params["model"]["encoding"]["n_levels"], 
        repr_casted->params["model"]["encoding"]["n_features_per_level"], 
        repr_casted->params["model"]["encoding"]["log2_hashmap_size"],
        repr_casted->params["model"]["encoding"]["base_resolution"], 
        repr_casted->params["model"]["encoding"]["per_level_scale"], 
        repr_casted->params["model"]["network"]["activation"], 
        repr_casted->params["model"]["network"]["output_activation"]
    );
    init_weights(model, repr_casted->weights);
  
    std::cout<<"entering decodeVNR 2"<<std::endl;
    
    std::cout<<"might here!"<<std::endl;

    // auto input_to_grid = py::module_::import("dvnr").attr("input_to_grid");

    // dvnr::vec3i dims = dvnr::vec3i(repr_casted->params["volume"]["dims"]["x"],repr_casted->params["volume"]["dims"]["y"],repr_casted->params["volume"]["dims"]["z"]);
    
    // py::object matrix = input_to_grid(dims);
    
    // torch::Tensor matr = matrix.cast<torch::Tensor>();
    
    torch::Tensor result;
    // model.zero_grad();
    // result = model.forward(matr);
    int dims[3];
    dims[0] = repr_casted->params["macrocell"]["dims"]["x"];
    dims[0]+=1;
    dims[1] = repr_casted->params["macrocell"]["dims"]["y"];
    dims[1]+=1;
    dims[2] = repr_casted->params["macrocell"]["dims"]["z"];
    dims[2]+=1;
    std::cout <<"xdimension is" <<repr_casted->params["macrocell"]["dims"]["x"] << std::endl;
    // double coords = new double[dims[0] * dims[1] * dims[2]*3];
    std::vector<double> coords;
    // coords.clear();
    // memset(coords, 0, sizeof(coords));
    // coords.reserve(3*dims[0] * dims[1] * dims[2]);
    // int i = 0;
    model.zero_grad();
    std::cout<<"before loop"<<std::endl;
    for(int x=0; x< dims[0]; x++){
        for(int y=0; y<dims[1];y++){
            for(int z=0;z<dims[2];z++){
                // std::vector<double> temp;
                double j = x/(dims[0]);
                coords.push_back(j);
                double k = y/(dims[1]);
                coords.push_back(k);
                double l = z/(dims[2]);
                coords.push_back(l);
            }
        }
    }
    std::cout<< "before blob" <<std::endl;
    auto tcoords = torch::tensor(coords);
    tcoords = tcoords.reshape({dims[0]*dims[1]*dims[2], 3});
    std::cout<< "before forward" << tcoords.sizes() <<std::endl;
    result = model.forward(tcoords);
    result = result.reshape({dims[2],dims[1],dims[0]});

    int64_t num_elements = result.numel();

    int64_t element_size = result.element_size();

    int64_t total_size_in_bytes = num_elements * element_size;

    
    int fd = open(filename, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    write(fd, result.data_ptr(), total_size_in_bytes);
    close(fd);
    // free(repr_casted);
    return;

}


}

