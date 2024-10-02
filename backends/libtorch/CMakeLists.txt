
add_library(dvnr_lib SHARED 
	backend.cpp
	networks.cpp
)
target_include_directories(dvnr_lib PUBLIC $<BUILD_INTERFACE:${DVNR_ROOT_DIR}>)
target_include_directories(dvnr_lib PUBLIC $<INSTALL_INTERFACE:include>)

target_compile_definitions(dvnr_lib INTERFACE ENABLE_DVNR_LIB_BACKEND)
target_compile_definitions(dvnr_lib PRIVATE ${DVNR_GLIBCXX_CXX11_ABI})

target_link_libraries(dvnr_lib PRIVATE 
	# libtorch 
	# gdt 
	util # instantvnr
	pybind11::module 
	"${TORCH_LIBRARIES}"
)
