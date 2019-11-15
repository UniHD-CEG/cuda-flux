#include <iostream>
#include <fstream>
#include <chrono>

__global__
void saxpy(int n, float a, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
    y[i] = a * x[i] + y[i];
}

using namespace std;
using namespace chrono;

#define CU_CHK(ERRORCODE) \
{cudaError_t error = ERRORCODE; \
  if (error != 0) \
  { cerr << cudaGetErrorName(error) << ": " << cudaGetErrorString(error) << \
    " at " << __FILE__ << ":" << __LINE__ << "\n";}}

int main(int argc, char **argv) {
  int N = 1 * (1 << 20);
  float *x, *y, *res, *d_x, *d_y;
  x = (float *) malloc(N * sizeof(float));
  y = (float *) malloc(N * sizeof(float));
  res = (float *) malloc(N * sizeof(float));

  CU_CHK(cudaMalloc(&d_x, N * sizeof(float)));
  CU_CHK(cudaMalloc(&d_y, N * sizeof(float)));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f + float(i);
  }

  cout << "CTAs: " << (N + 511) / 512 << "\n";

  auto t0 = high_resolution_clock::now();
  CU_CHK(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
  CU_CHK(cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice));
  auto t1 = high_resolution_clock::now();

  saxpy <<< (N + 511) / 512, 512 >>> (N, 2.0f, d_x, d_y);
  CU_CHK(cudaGetLastError());
  CU_CHK(cudaDeviceSynchronize());
  auto t2 = high_resolution_clock::now();

  CU_CHK(cudaMemcpy(res, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));
  auto t3 = high_resolution_clock::now();

  if (cudaGetLastError() != 0)
    return -1;

  for (int i = 0; i < N; i++) {
    float y_host = 2.0f * x[i] + y[i];
    float diff = y_host - res[i];
    if (diff > 1.0f)
      cout << "Error at y[" << i << "]: " << y_host << " vs. " << res[i] << "\n";
  }

  cout << "Duaration memcpy to device: " << duration_cast<microseconds>(t1 - t0).count() << endl;
  cout << "Duaration kernel: " << duration_cast<microseconds>(t2 - t1).count() << endl;
  cout << "Duaration memcpy to host: " << duration_cast<microseconds>(t3 - t1).count() << endl;

  // write to file
  if (argc > 1) {
    ofstream file(argv[1], std::ios::binary);
    file.write((char *) y, N * sizeof(float));
  }

  return 0;
}
