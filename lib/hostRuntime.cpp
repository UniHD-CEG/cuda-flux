#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <sys/time.h>

using namespace std;

// TODO implement logging/log levels

#define CUDA_CHECK(ERRORCODE)                                                  \
  {                                                                            \
    cudaError_t error = ERRORCODE;                                             \
    if (error != 0) {                                                          \
      cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error)     \
           << " at " << __FILE__ << ":" << __LINE__ << "\n";                   \
    }                                                                          \
  }

extern "C" uint64_t *__attribute__((weak)) createBBCounterMemory(size_t len) {
  // cerr << "Initializing BB Counter Memory...\n";

  uint64_t *d_ptr;

  CUDA_CHECK(cudaMalloc(&d_ptr, len * sizeof(uint64_t)));

  if(d_ptr == nullptr) {
    cerr << "Error allocating memory for basic block counters. Exiting...\n";
    exit(-1);
  }


  CUDA_CHECK(cudaMemset(d_ptr, 0, len * sizeof(uint64_t)));

  return d_ptr;
}

__attribute__((weak)) bool g_profilingModePolled = false;
__attribute__((weak)) uint32_t g_profilingMode = 0;

extern "C" uint32_t __attribute__((weak)) getProfilingMode() {
  // cerr << "ProfilingModePolled " << g_profilingModePolled << "\n";
  // cerr << "ProfilingMode " << g_profilingMode << "\n";

  if (g_profilingModePolled)
    return g_profilingMode;

  char *profilingMode;
  profilingMode = getenv("MEKONG_PROFILINGMODE");
  if (profilingMode != NULL) {
    g_profilingMode = atoi(profilingMode);
    g_profilingModePolled = true;
  }

  // cerr << "ProfilingModePolled " << g_profilingModePolled << "\n";
  // cerr << "ProfilingMode " << g_profilingMode << "\n";

  return g_profilingMode;
}

extern "C" struct timeval *__attribute__((weak)) startClock() {
  struct timeval *t = new struct timeval();
  gettimeofday(t, nullptr);
  return t;
}

// TODO print when debug env is set
extern "C" void __attribute__((weak))
serializeBBCounter(uint64_t *d_blockList, size_t len, uint32_t n_banks,
                   const char *kernelname, uint64_t gridXY, uint32_t gridZ,
                   uint64_t blockXY, uint32_t blockZ, uint64_t shmsize,
                   cudaStream_t stream, struct timeval *t1) {
  // cerr << "Serializing BB Counters of kernel " << string(kernelname) << "
  // ...\n";
  CUDA_CHECK(cudaStreamSynchronize(stream));

  struct timeval t2;
  gettimeofday(&t2, nullptr);

  unsigned long diff =
      (t2.tv_sec - (t1->tv_sec)) * 10e6 + (t2.tv_usec - (t1->tv_usec));
  free(t1);
  // No need to free t2 its not a pointer
  // free(t2);

  uint64_t *h_blockList = (uint64_t *)malloc(len * sizeof(uint64_t));

  // Copy basic block counter to host memory
  CUDA_CHECK(cudaMemcpy(h_blockList, d_blockList, len * sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));

  for (int block_ID = 0; block_ID < len / n_banks; ++block_ID) {
    uint64_t sum = 0;
    for (int bank = 0; bank < n_banks; ++bank) {
      if (__builtin_add_overflow(sum, h_blockList[block_ID * n_banks + bank],
                                 &sum)) {
        // todo
        cerr << "CUDA FLUX: OVERFLOW!\n";
      }
      // sum += h_blockList[block_ID * n_banks + bank];
    }

    // Store result to column for bank zero
    h_blockList[block_ID * n_banks + 0] = sum;
  }

  ofstream output("bbc.txt", ios::app);

  // TODO dump version
  if (output.tellp() == 0) {
    output << "# CUDA_Flux_Version=0.3\n";
  }

  output << string(kernelname) << ", "
         << (uint32_t)(gridXY & 0x00000000FFFFFFFF) << ", "
         << (uint32_t)(((uint64_t)gridXY & 0xFFFFFFFF00000000) >> 32) << ", "
         << gridZ << ", " << (uint32_t)(blockXY & 0x00000000FFFFFFFF) << ", "
         << (uint32_t)(((uint64_t)blockXY & 0xFFFFFFFF00000000) >> 32) << ", "
         << blockZ << ", " << shmsize << ", " << diff << ", "
         << g_profilingMode;

  for (int block_ID = 0; block_ID < len / n_banks; ++block_ID) {
    // Read result from column for bank zero
    output << ", " << h_blockList[block_ID * n_banks + 0];
  }

  output << "\n";
  output.flush();

  CUDA_CHECK(cudaFree(d_blockList));
}

extern "C" void __attribute__((weak))
mekong_WritePTXBlockAnalysis(bool *isWritten, const char *analysis) {

  if (*isWritten)
    return;

  // cerr << "Serializing PTX Analysis...\n";

  ofstream output("PTX_Analysis.yml", ios::app);

  output << analysis;

  *isWritten = true;
}
