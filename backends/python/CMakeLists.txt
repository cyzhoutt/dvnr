# -----------------------------
# referencing torch
# -----------------------------

# execute_process(
#   COMMAND "${PYTHON_EXECUTABLE}" -c 
#     "import torch, os; print(os.path.dirname(torch.__file__))"
#   OUTPUT_VARIABLE PYTHON_TORCH_SITE
#   OUTPUT_STRIP_TRAILING_WHITESPACE
# )

# add_library(libtorch INTERFACE)

# target_link_directories(libtorch INTERFACE
# 	${PYTHON_TORCH_SITE}/lib
# )
# target_include_directories(libtorch INTERFACE 
# 	${PYTHON_TORCH_SITE}/include
# 	${PYTHON_TORCH_SITE}/include/torch/csrc/api/include
# 	${PYTHON_TORCH_SITE}/include/TH
# 	${PYTHON_TORCH_SITE}/include/THC
# )

# find_library(LibTorch_torch_cuda_LIBRARY     NAMES torch_cuda     PATHS ${PYTHON_TORCH_SITE}/lib)
# find_library(LibTorch_torch_cuda_cu_LIBRARY  NAMES torch_cuda_cu  PATHS ${PYTHON_TORCH_SITE}/lib)
# find_library(LibTorch_torch_cuda_cpp_LIBRARY NAMES torch_cuda_cpp PATHS ${PYTHON_TORCH_SITE}/lib)

# target_link_libraries(libtorch INTERFACE 
#   c10 c10_cuda
#   torch torch_python 
#   torch_cpu
#   # torch_cuda 
#   # torch_cuda_cu torch_cuda_cpp
# )
# if (LibTorch_torch_cuda_LIBRARY)
#   target_link_libraries(libtorch INTERFACE ${LibTorch_torch_cuda_LIBRARY})
# endif()
# if (LibTorch_torch_cuda_cu_LIBRARY)
#   target_link_libraries(libtorch INTERFACE ${LibTorch_torch_cuda_cu_LIBRARY})
# endif()
# if (LibTorch_torch_cuda_cpp_LIBRARY)
#   target_link_libraries(libtorch INTERFACE ${LibTorch_torch_cuda_cpp_LIBRARY})
# endif()

# target_compile_options(libtorch INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

# -----------------------------
#
# -----------------------------

add_library(dvnr_python SHARED 
	backend.cpp
	backend.cu
)
target_include_directories(dvnr_python PUBLIC $<BUILD_INTERFACE:${DVNR_ROOT_DIR}>)
target_include_directories(dvnr_python PUBLIC $<INSTALL_INTERFACE:include>)
target_link_libraries(dvnr_python PRIVATE 
	util sampler
	pybind11::embed
  	"${TORCH_LIBRARIES}"
)
target_compile_definitions(dvnr_python INTERFACE ENABLE_DVNR_PYTHON_BACKEND)
target_compile_definitions(dvnr_python PRIVATE ${DVNR_GLIBCXX_CXX11_ABI})

# set_target_properties(dvnr_python PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
# set_target_properties(dvnr_python PROPERTIES CUDA_SEPARABLE_COMPILATION  ON)


# -----------------------------
#
# -----------------------------

pybind11_add_module(dvnr_ext
	bindings.cpp 
)
target_include_directories(dvnr_ext PRIVATE $<BUILD_INTERFACE:${DVNR_ROOT_DIR}>)
target_link_libraries(dvnr_ext PRIVATE 
	# libtorch 
	# gdt 
	util # instantvnr
	pybind11::module 
	"${TORCH_LIBRARIES}"
)
