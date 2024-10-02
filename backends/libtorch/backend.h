// #include <dvnr/dvnr_internal.h>
#include <string>
#include "dvnr/dvnr_internal.h"

#include "networks.hpp"


using namespace std;
using INR_Base = INR_Cpp;

namespace dvnr {
namespace lib_backend {

nlohmann::json create_inr_scene(const std::string& fn, dvnr::vec3i& dims, INR_Base& model, nlohmann::json& encoding_config, nlohmann::json& network_config);
struct McReturn{
    dvnr::vec3i volumedims;
    dvnr::vec3i dims;
    dvnr::vec3f spacings;
    std::vector<float> value_ranges;
};
McReturn compute_macrocell(dvnr::vec3i& vdims, INR_Base& model, int mcsize_mip = 4);
void save_json_binary(const nlohmann::json& root, std::string filename);
torch::Tensor generate_grid(int dimx, int dimy, int dimz, torch::Device device);
void init_weights(torch::nn::Module& network, py::dict weights);

    void initialize();
    void finalize();
    dvnr_t createVNR(const DataDesc& desc, const ModelDesc& modelParam, const OptimizerDesc& descoptim);
    void decodeVNR(const dvnr_t& repr, float* data);
    void decodeVNR(const dvnr_t& repr, const char* filename);

}
}