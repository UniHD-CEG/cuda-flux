#include <iostream>
#include <fstream>
#include <chrono>
#include <random>

using namespace std;
using namespace chrono;

#define CU_CHK(ERRORCODE) \
{cudaError_t error = ERRORCODE; \
  if (error != 0) \
  { cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << \
    " at " << __FILE__ << ":" << __LINE__ << "\n";}}

__global__
void branch_div_test(int n, int *in, float *out) {
  // Execute sum_thread times
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if ( id >= n)
    return;  // execute sum_thread - n times


  // execute n times
  float sum = 0.0;
  for (int i = 0; i<in[id]; ++i) {
    // execute sum(in) times
    if ( i % 7 == 0)
      sum -= 1.1; // execute sum(in) / 7 times
    else
      sum += 2; // execute sum(in) - sum(in) / 7
  }

  out[id] = sum; // execute n times
}

int main(int argc, char **argv) {
  int n = 512;
  if (argc > 1)
    n = atoi(argv[1]);

  int *in = (int *) malloc(n * sizeof(int));
  float *out = (float *) malloc(n * sizeof(float));

  int *d_in;
  float *d_out;

  CU_CHK(cudaMalloc(&d_in, n * sizeof(int)));
  CU_CHK(cudaMalloc(&d_out, n * sizeof(float)));

  // init host memory
  auto rand = minstd_rand();
  long sum = 0;
  for (int i = 0; i < n; ++i) {
    int tmp = rand()/1000000;
    sum += tmp;
    in[i] = tmp;
  }
  memset(out, 0, n);

  // init device memory
  CU_CHK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));
  cudaMemset ( d_out, 0, n);


  auto t0 = high_resolution_clock::now();
  branch_div_test <<< (n+(16-1))/16, 16 >>> (n, d_in, d_out);
  CU_CHK(cudaGetLastError());
  CU_CHK(cudaDeviceSynchronize());
  auto t1 = high_resolution_clock::now();

  CU_CHK(cudaMemcpy(out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

  if (cudaGetLastError() != 0)
    return -1;

  cout << "Duaration kernel: " << duration_cast<microseconds>(t1 - t0).count() << endl;
  cout << "N, sum:\n" << n << ", " << sum << "\n";
  //cout << (n/16)*16 << ", " << n << ", " << sum << ", " << sum/7 << ", " << sum - (sum/7) << endl;
  //cout << sum << endl;
  //cout << sizeof(long) << "\t" << sizeof(uint64_t) << endl;

  return 0;
}
