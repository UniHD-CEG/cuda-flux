# CUDA Flux: A Lightweight Instruction Profiler for CUDA Applications

CUDA Flux is a profiler for GPU applications which reports the basic block executions frequencies of compute kernels

# Dependencies

* LLVM:  
  Git commit 4c9d0da8382f176a2fb7b97298932a53d22e8627 from https://github.com/llvm/llvm-project/
  CMake config:
  ```
  cmake -DLLVM_ENABLE_PROJECTS=clang -DCMAKE_INSTALL_PREFIX=/opt/llvm-master -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_ENABLE_DOXYGEN=OFF -DLLVM_BUILD_DOCS=OFF -GNinja ../llvm
  ```
* re2c lexer generator - http://re2c.org/ (make sure to check your package manager first)
* CUDA SDK 8.0  
  Newer version are currently not supported due to different kernel launch implementation.
* Python (python3 preferred) with the yaml package installed
* environment-modules (optional, but recommended)

## Install

```
# Make sure LLVM and CUDA are in your paths
git clone https://github.com/UniHD-CEG/cuda-flux.git
cd cuda-flux && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/cuda-flux .. # change install dir if you wish
make install
```

## Usage

Make sure the bin folder of your cuda-flux installation is in your path.
Better: use environment-module to load cuda-flux.
```
module load /opt/cuda-flux/module/cuda_flux # LLVM and CUDA need to be loaded first!
```

Compile your CUDA application:

```
clang_cf++ --cuda-gpu-arch=sm_30 -std=c++11 -lcudart test/saxpy.cu -o saxpy`
```

Execute the binary like usual: `./saxpy`


Output:  
If any kernel was executed there is a bbc.txt file. Each kernel execution
results in a line with the following information:
* kernel name
* gridDim{X,Y,Z}
* blockDim{X,Y,Z}
* shared Memory size
* Profiling Mode (full, CTA, warp)
* List of executions counters for each block  

The PTX instructions for all kernels are stored in the `PTX_Analysis.yml`
file. The order of the Basic Blocks corresponds with the order of the 
counters in the bbc.txt file.
