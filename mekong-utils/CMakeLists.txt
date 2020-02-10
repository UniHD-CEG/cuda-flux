cmake_minimum_required(VERSION 3.6)

# Build ptx parser header file
add_custom_command( OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/inc/ptx_parser.h
                    COMMAND re2c -o inc/ptx_parser.h inc/ptx_parser.hre
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    DEPENDS inc/ptx_parser.hre )
add_custom_target( ptx_parser DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/inc/ptx_parser.h)

# If Standalone project
if ( CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR )
  project(mekong-utils)

  find_package(LLVM REQUIRED CONFIG)
  add_definitions(${LLVM_DEFINITIONS})
  include_directories(${LLVM_INCLUDE_DIRS})
  link_directories(${LLVM_LIBRARY_DIRS})
endif()

set(LLVM_ENABLE_PLUGINS ON)

add_library( mekong-utils STATIC 
  src/DeviceUtils.cpp 
  src/HostUtils.cpp 
  src/IRUtils.cpp
  src/BasicBlockEnumerator.cpp)
add_dependencies( mekong-utils ptx_parser)
#target_link_libraries( mekong-utils PUBLIC LLVMTransformUtils)
target_include_directories(mekong-utils PUBLIC inc)

set_target_properties(mekong-utils PROPERTIES COMPILE_FLAGS "-fno-rtti" POSITION_INDEPENDENT_CODE ON)

# Get proper shared-library behavior (where symbols are not necessarily
# resolved when the shared library is linked) on OS X.
if(APPLE)
	set_target_properties(mekong-utils PROPERTIES
        LINK_FLAGS "-undefined dynamic_lookup"
    )
endif(APPLE)
