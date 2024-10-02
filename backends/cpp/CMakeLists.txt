# -----------------------------
#
# -----------------------------
if(NOT ENABLE_CUDA)
	return()
endif()

add_library(dvnr_cpp SHARED 
	backend.cpp
	backend.cu
)
target_include_directories(dvnr_cpp PUBLIC $<BUILD_INTERFACE:${DVNR_ROOT_DIR}>)
target_include_directories(dvnr_cpp PUBLIC $<INSTALL_INTERFACE:include>)
target_link_libraries(dvnr_cpp PRIVATE 
	gdt util instantvnr
)
target_compile_definitions(dvnr_cpp INTERFACE ENABLE_DVNR_CPP_BACKEND)
target_compile_definitions(dvnr_cpp PRIVATE ${DVNR_GLIBCXX_CXX11_ABI})
set_target_properties(dvnr_cpp PROPERTIES 
	CUDA_RESOLVE_DEVICE_SYMBOLS ON
	CUDA_SEPARABLE_COMPILATION  ON
)
