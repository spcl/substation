#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void wait_kernel(long long int cycles) {
  const long long int start = clock64();
  long long int cur;
  do {
    cur = clock64();
  } while (cur - start < cycles);
}

}  // anonymous namespace

/**
 * Launch a kernel on stream that waits for length seconds.
 */
void gpu_wait(double length, cudaStream_t stream) {
  // Estimate GPU frequency to convert seconds to cycles.
  static long long int freq_hz = 0;  // Cache.
  if (freq_hz == 0) {
    int device;
    cudaGetDevice(&device);
    int freq_khz;
    cudaDeviceGetAttribute(&freq_khz, cudaDevAttrClockRate, device);
    freq_hz = (long long int) freq_khz * 1000;   // Convert from KHz.
  }
  double cycles = length * freq_hz;
  wait_kernel<<<1, 1, 0, stream>>>((long long int) cycles);
}
