cmake_minimum_required(VERSION 3.6)

project("CUDA_Flux")

find_package(LLVM REQUIRED CONFIG)
add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

set(LLVM_ENABLE_PLUGINS ON)

add_subdirectory(mekong-utils)

add_custom_command( OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/inc/deviceRuntime.h
	COMMAND clang++  -S -emit-llvm --cuda-device-only --cuda-gpu-arch=sm_30 -O3 -std=c++11
	${CMAKE_CURRENT_SOURCE_DIR}/lib/deviceRuntime.cu -o deviceRuntime.ll
	COMMAND xxd -i deviceRuntime.ll > ${CMAKE_CURRENT_SOURCE_DIR}/inc/deviceRuntime.h
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/deviceRuntime.cu )

add_custom_target( deviceRuntime DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inc/deviceRuntime.h)

add_custom_command( OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/inc/hostRuntime.h
	COMMAND clang++ -S -emit-llvm -O3
	${CMAKE_CURRENT_SOURCE_DIR}/lib/hostRuntime.cpp -o hostRuntime.ll
	COMMAND xxd -i hostRuntime.ll > ${CMAKE_CURRENT_SOURCE_DIR}/inc/hostRuntime.h
	DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/hostRuntime.cpp )

add_custom_target( hostRuntime DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inc/hostRuntime.h)

add_library(cuda_flux_pass MODULE
  # List your source files here.
  lib/fluxDevicePass.cpp
  lib/fluxHostPass.cpp
  lib/registerPasses.cpp)

# TODO implement mekong-utils dependency properly
target_include_directories(cuda_flux_pass PRIVATE inc mekong-utils/inc) 

add_dependencies( cuda_flux_pass deviceRuntime hostRuntime)
target_link_libraries( cuda_flux_pass PRIVATE mekong-utils)

set_target_properties(cuda_flux_pass PROPERTIES COMPILE_FLAGS "-fno-rtti" POSITION_INDEPENDENT_CODE ON)

set_target_properties( cuda_flux_pass PROPERTIES LINK_FLAGS "-Xlinker -zdynamic-undefined-weak")
# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
	set_target_properties(cuda_flux_pass PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)

# Testing
if(EXISTS ${LLVM_MAIN_SRC_DIR}/utils/lit/lit.py)
	message("lit found")
else()
	message("lit not found")
endif()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#	message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

add_test( NAME Saxpy
        COMMAND clang++ --cuda-gpu-arch=sm_30 -std=c++11 -lcudart
        -finline-functions
        -Xclang -load -Xclang ${CMAKE_CURRENT_BINARY_DIR}/libcuda_flux_pass.so
        ${CMAKE_CURRENT_SOURCE_DIR}/test/saxpy.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/saxpy )
add_test( NAME BranchDivergence
	COMMAND clang++ --cuda-gpu-arch=sm_30 -std=c++11 -lcudart
	-finline-functions
	-Xclang -load -Xclang ${CMAKE_CURRENT_BINARY_DIR}/libcuda_flux_pass.so
	${CMAKE_CURRENT_SOURCE_DIR}/test/branchdivergence.cu
	-o ${CMAKE_CURRENT_BINARY_DIR}/BranchDivergence )
# Regression tests
add_test(NAME rodinia_hotspot 
	COMMAND opt -load ${CMAKE_CURRENT_BINARY_DIR}/libcuda_flux_pass.so -S --host-cuda-flux rodinia_hotspot_hostcode.bc
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/regression_test_ir)
add_test(NAME rodinia_nn
	COMMAND opt -load ${CMAKE_CURRENT_BINARY_DIR}/libcuda_flux_pass.so -S --host-cuda-flux rodinia_nn_hostcode.bc
	WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/regression_test_ir)
add_test(NAME MyTest COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/runtime_test.sh)
set_tests_properties(MyTest PROPERTIES SKIP_RETURN_CODE 77)

enable_testing()

# Install
configure_file(utils/modulefile cuda_flux)
configure_file(utils/clang_cf++ clang_cf++)
configure_file(utils/parseRegisterUsage.py parseRegisterUsage.py)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cuda_flux DESTINATION ${CMAKE_INSTALL_PREFIX}/module)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/clang_cf++ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin PERMISSIONS
  OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/parseRegisterUsage.py DESTINATION ${CMAKE_INSTALL_PREFIX}/bin PERMISSIONS
  OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ)
install(TARGETS cuda_flux_pass DESTINATION lib)
