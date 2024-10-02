#include <dvnr/dvnr.h>
// #include <dvnr/instantvnr/core/sampler.h>

#include <iostream>
#include <functional>
#include <fstream>

#include <memory>

using namespace dvnr;

std::string tfn = "data/visualization.json";

VolumeDesc_Structured data_engine()
{
    VolumeDesc_Structured desc;

    desc.shape.dimx = 256;
    desc.shape.dimy = 256;
    desc.shape.dimz = 128;

    desc.shape.dtype = "uint8";
    desc.offset = 0;

    desc.filename = "/Users/adazhou/Desktop/dvnr/distributed-vnr/data/engine_256x256x128_uint8.raw";

    // desc.filename = "/Users/qwu/Work/projects/distributed-vnr/data/engine_256x256x128_uint8.raw";

    desc.is_big_endian = false;

    desc.shape.min = 0;
    desc.shape.max = 255;

    // tfn = "data/visualization_1atmhr.json";
    return desc;
}

VolumeDesc_Structured data_1atm_temp_4x()
{
    VolumeDesc_Structured desc;

    desc.shape.dimx = 864;
    desc.shape.dimy = 240;
    desc.shape.dimz = 640;

    desc.shape.dtype = "float32";
    desc.offset = 0;
    desc.filename = "data/datasets/1atm_temp_4.1000E-04__864x240x640_float32.raw";
    desc.is_big_endian = false;

    desc.shape.min = 6.2369;
    desc.shape.max = 15.2148;

    tfn = "data/visualization_1atmtemp.json";
    return desc;
}


VolumeDesc_Structured data_mechhand()
{
    VolumeDesc_Structured desc;

    desc.shape.dimx = 640;
    desc.shape.dimy = 220;
    desc.shape.dimz = 229;

    desc.shape.dtype = "float32";
    desc.offset = 0;
    desc.filename = "data/datasets/MechHand_f_640x220x229.raw";
    desc.is_big_endian = false;

    desc.shape.min = 0;
    desc.shape.max = 0.96428573131561279;

    tfn = "data/visualization_mechhand.json";

    return desc;
}

dvnr_t example1(VolumeDesc_Structured desc)
{
    const size_t size = (size_t)desc.shape.dimx*(size_t)desc.shape.dimy*(size_t)desc.shape.dimz;
    std::shared_ptr<char[]> buffer(new char[size * sizeof(float)]);

    desc.dst = (void*)buffer.get();

    dvnrLoadData(desc);

    DataDesc data = desc.shape;

    
    data.fields = { (void*)buffer.get() };
    // data.fields_flatten = (void*)buffer.get();
    data.fields.push_back(desc.dst);
    data.fields_flatten = desc.dst;

    data.enable_clipping = true;
    data.clipbox[0] = data.dimx/4;
    data.clipbox[1] = data.dimy/4;
    data.clipbox[2] = 0;
    data.clipbox[3] = data.dimx/4*3;
    data.clipbox[4] = data.dimy/4*3;
    data.clipbox[5] = data.dimz;

    ModelDesc model;
    model.tcnn = true;
    model.n_hidden_layers = 2;
    model.n_neurons = 64;
    model.n_levels = 8;
    model.n_features_per_level = 4;

    OptimizerDesc optimizer;
    optimizer.max_steps = 10000;
    // optimizer.psnr_target = 30;
    optimizer.verbose = true;

    return dvnrCreate(data, model, optimizer);
}

int main() 
{
    // dvnrInitialize("python");
    // dvnrInitialize("cpp");
    dvnrInitialize("lib");

    VolumeDesc_Structured desc = data_engine();
    printf("Main here! 0");
    auto nv = example1(desc);
    printf("Main here! 1");
    dvnrDecode(nv, "finalFile.raw");
    dvnrFree(nv);
    dvnrFinalize();

    // dvnrGPUMemory_FreeTemporary();
    // dvnrGPUMemory_PrintUsage("[dvnr]"); // Optional

    return 0;
}
