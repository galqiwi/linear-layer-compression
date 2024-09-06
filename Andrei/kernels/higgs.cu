#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "grids.h"

template<int group_size=2, int codebook_bits=8, int hadamard_size=1024>
__global__ void HiggsAlignedMatVec(
  const uint4* __restrict__ A,
  const uint4* __restrict__ B,
        uint4* __restrict__ C,
  const half* __restrict__ scales,
  int prob_m,
  int prob_k
) {
  constexpr int halfs_in_uint4 = 128 / 16;
  constexpr int threads_in_wave = 32;
  constexpr int steps_in_wave = hadamard_size / (threads_in_wave * 8);

  constexpr int codebook_size = group_size * (1 << codebook_bits) / halfs_in_uint4;
  __shared__ uint4 sh_codebook[codebook_size];
  for (int i = threadIdx.x; i < codebook_size; i += blockDim.x) {
    if constexpr (group_size == 2 && codebook_bits == 8) {
      sh_codebook[i] = HIGGS_2_256[i];
    } else if constexpr (group_size == 4 && codebook_bits == 8) {
      sh_codebook[i] = HIGGS_4_256[i];
    }
  }

  int a_gl_stride = prob_k / group_size / (16 / codebook_bits) / halfs_in_uint4;
  int a_gl_rd = (blockDim.x / threads_in_wave) * blockIdx.x + (threadIdx.x / threads_in_wave);
  bool pred = a_gl_rd < prob_m;
  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % threads_in_wave;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % threads_in_wave;

  __shared__ uint4 sh_b[threads_in_wave * (steps_in_wave + 1)];
  float res = 0;

  int iters = (prob_k - 1) / hadamard_size + 1;
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < threads_in_wave * steps_in_wave; i += blockDim.x) {
      if (steps_in_wave * (b_gl_rd + i) < prob_k)
        sh_b[(steps_in_wave + 1) * (i / steps_in_wave) + i % steps_in_wave] = B[b_gl_rd + i];
    }
    __syncthreads();

    float iter_res = 0;

    int b_sh_rd = (steps_in_wave + 1) * (threadIdx.x % threads_in_wave);
    if (pred && a_gl_rd < a_gl_end) {
      float scale = __half2float(scales[a_gl_rd * halfs_in_uint4 * (16 / codebook_bits) * group_size / hadamard_size]);
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
            
      #pragma unroll
      for (int i = 0; i < steps_in_wave; i++) {
        uint32_t dec[4];
        #pragma unroll
        for (int j = 0; j < 8 / group_size; j++) {
          if constexpr (group_size == 2 && codebook_bits == 8) {
            ((uint32_t*)dec)[j] = ((uint32_t*)sh_codebook)[enc[(8 / group_size) * i + j]]; // read 2 halfs at a time
          } else if constexpr (group_size == 4 && codebook_bits == 8) {
            ((uint64_t*)dec)[j] = ((uint64_t*)sh_codebook)[enc[(8 / group_size) * i + j]]; // read 4 halfs at a time
          }
        }

        if (threadIdx.x == 31) {
          printf("a_gl_rd: %i, scale_offset_fp16: %i, scale: %f, second_code: %i, last_decoded: %f, last_b: %f, iter_res: %f\n",
            a_gl_rd,
            a_gl_rd * halfs_in_uint4 * (16 / codebook_bits) * group_size / hadamard_size,
            scale,
            enc[(8 / group_size) * i + 1],
            __half2float(((half*)dec)[7]),
            __half2float(reinterpret_cast<half*>(&sh_b[b_sh_rd])[7]),
            iter_res
          );
        }
        
        half2* a = reinterpret_cast<half2*>(&dec);
        half2* b = reinterpret_cast<half2*>(&sh_b[b_sh_rd]);
        half2 res2 = {};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          res2 = __hfma2(a[j], b[j], res2);
        }
        iter_res += __half2float(res2.x) + __half2float(res2.y);
        b_sh_rd += 1;
      }
      iter_res *= scale;
      a_gl_rd += threads_in_wave * (16 / codebook_bits) / group_size;
    }
    b_gl_rd += threads_in_wave * steps_in_wave; // Move by hadamard_size
    res += iter_res;
  }

  if (pred) {
    #pragma unroll
    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % threads_in_wave == 0) {
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
    }
  }
}

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

const int THREAD_M = 16;


template<int group_size, int codebook_bits, int hadamard_size>
void  higgs_aligned_matvec_cuda(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ scales,
  int prob_m,
  int prob_k
) {
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, 0);

  int sms;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  int waves = 0;
  int thread_m;
  do {
    waves++;
    thread_m = ceildiv(prob_m, waves * sms);
  } while (thread_m > THREAD_M);

  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  int shared = 16 * 32 * (hadamard_size / 32 / 8 + 1) + 16 * group_size * (1 << codebook_bits);
  cudaFuncSetAttribute(
    HiggsAlignedMatVec<group_size, codebook_bits, hadamard_size>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
  );
  HiggsAlignedMatVec<group_size, codebook_bits, hadamard_size><<<blocks, threads, shared>>>(
    (const uint4*) A,
    (const uint4*) B,
    (uint4*) C,
    (const half*) scales,
    prob_m,
    prob_k
  );
}

template void  higgs_aligned_matvec_cuda<2, 8, 1024>(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ scales,
  int prob_m,
  int prob_k
);

template void  higgs_aligned_matvec_cuda<4, 8, 1024>(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ scales,
  int prob_m,
  int prob_k
);

// #define CUDACHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
// inline void cuda_check(cudaError_t error_code, const char *file, int line)
// {
//     if (error_code != cudaSuccess)
//     {
//         fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
//         fflush(stderr);
//         exit(error_code);
//     }
// }

// int main() {
//     const auto codes = std::vector<uint8_t>(1024 * 1024 / 2, 1);
//     uint8_t* codes_device;
//     cudaMalloc(&codes_device, 1024 * 1024 / 2);
//     cudaMemcpy((void**)codes_device, codes.data(), 1024 * 1024 / 2, cudaMemcpyHostToDevice);
    
//     const auto scales = std::vector<__half>(1024, 0.0001);
//     __half* scales_device;
//     cudaMalloc(&scales_device, 1024 * 2);
//     cudaMemcpy((void**)scales_device, scales.data(), 1024 * 2, cudaMemcpyHostToDevice);

//     const auto input = std::vector<__half>(1024, 1);
//     __half* input_device;
//     cudaMalloc(&input_device, 1024 * 2);
//     cudaMemcpy((void**)input_device, input.data(), 1024 * 2, cudaMemcpyHostToDevice);

//     auto output = std::vector<__half>(1024, 0);
//     __half* output_device;
//     cudaMalloc(&output_device, 1024 * 2);
//     cudaMemcpy((void**)output_device, output.data(), 1024 * 2, cudaMemcpyHostToDevice);

//     higgs2x256_matvec_cuda(
//         codes_device,
//         input_device,
//         output_device,
//         scales_device,
//         1024,
//         1024
//     );
//     cudaMemcpy((void**)output.data(), output_device, 1024 * 2, cudaMemcpyDeviceToHost);
//     CUDACHECK(cudaPeekAtLastError());

//     std::cout << static_cast<float>(output[0]) << std::endl;

//     return 0;
// };
