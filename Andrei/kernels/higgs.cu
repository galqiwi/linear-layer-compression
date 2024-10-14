#include <iostream>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "grids.h"

template<int group_size, int codebook_bits, int scales_size>
__global__ void HiggsAlignedMatVec(
  const uint4* __restrict__ A,
  const uint4* __restrict__ B,
        uint4* __restrict__ C,
  const half* __restrict__ scales,
  int prob_m,
  int prob_k
) {
  constexpr int codes_in_half = 16 / codebook_bits;
  constexpr int halfs_in_uint4 = 128 / 16;
  constexpr int threads_in_wave = 32;
  constexpr int steps_in_wave = scales_size / (threads_in_wave * 8);

  int a_gl_stride_half = prob_k / group_size / codes_in_half;
  int c_gl_wr = (blockDim.x / threads_in_wave) * blockIdx.x + (threadIdx.x / threads_in_wave);
  bool pred = c_gl_wr < prob_m;
  int b_gl_rd = 0;
  int a_gl_rd_half = a_gl_stride_half * c_gl_wr + (threadIdx.x % threads_in_wave) * (scales_size / group_size) / threads_in_wave / codes_in_half;
  int a_gl_end_half = a_gl_stride_half * (c_gl_wr + 1);

  constexpr int replication = 1024 / 16 / group_size;
  int lane = threadIdx.x % replication;
  __shared__ uint4 sh_grid[replication * 256 * group_size / halfs_in_uint4];
  const uint4* codebook;
  if constexpr (group_size == 2 && codebook_bits == 8) {
    codebook = HIGGS_2_256;
  } else if constexpr (group_size == 4 && codebook_bits == 8) {
    codebook = HIGGS_4_256;
  } else if constexpr (group_size == 1 && codebook_bits == 8) {
    codebook = HIGGS_1_256;
  }
  
  for (int i = threadIdx.x; i < 256 * group_size / halfs_in_uint4; i += blockDim.x) {
    uint4 dec = codebook[i];
    #pragma unroll
    for (int j = 0; j < replication; j++)
      sh_grid[replication * i + (j + lane) % replication] = dec;
  }
  __syncthreads();

  __shared__ uint4 sh_b[threads_in_wave * (steps_in_wave + 1)];
  float res = 0;

  int iters = (prob_k - 1) / scales_size + 1;
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
    if (pred && a_gl_rd_half < a_gl_end_half) {
      float scale = __half2float(scales[a_gl_rd_half * codes_in_half * group_size / scales_size]);
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(reinterpret_cast<const half*>(A) + a_gl_rd_half);
            
      #pragma unroll
      for (int i = 0; i < steps_in_wave; i++) {
        uint32_t dec[4];
        #pragma unroll
        for (int j = 0; j < (8 - 1) / group_size + 1; j++) {
          if constexpr (group_size == 2 && codebook_bits == 8) {
            ((uint32_t*)dec)[j] = *(((const uint32_t*)sh_grid) + replication * enc[(8 / group_size) * i + j] + lane); // read 2 halfs at a time
          } else if constexpr (group_size == 4 && codebook_bits == 8) {
            ((uint64_t*)dec)[j] = *(((const uint64_t*)sh_grid) + replication * enc[(8 / group_size) * i + j] + lane); // read 4 halfs at a time
          } else if constexpr (group_size == 1 && codebook_bits == 8) {
            ((uint16_t*)dec)[j] = *(((const uint16_t*)sh_grid) + replication * enc[(8 / group_size) * i + j] + lane); // read 1 halfs at a time
          }
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
      a_gl_rd_half += scales_size / group_size / codes_in_half;
    }
    b_gl_rd += threads_in_wave * steps_in_wave; // Move by scales_size
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


template<int group_size, int codebook_bits, int scales_size>
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
  int shared = 16 * 32 * (scales_size / 32 / 8 + 1) + 16 * group_size * (1 << codebook_bits);
  cudaFuncSetAttribute(
    HiggsAlignedMatVec<group_size, codebook_bits, scales_size>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
  );
  HiggsAlignedMatVec<group_size, codebook_bits, scales_size><<<blocks, threads, shared>>>(
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

template void  higgs_aligned_matvec_cuda<1, 8, 1024>(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ scales,
  int prob_m,
  int prob_k
);

template<int group_size>
__global__ void HiggsKx256MatVec(
  const uint4* __restrict__ A,
  const uint4* __restrict__ B,
        uint4* __restrict__ C,
  const half* __restrict__ scales,
  int prob_m,
  int prob_k
) {
  constexpr int scales_size = 1024;
  constexpr int codebook_bits = 8;
  constexpr int steps_in_wave = 4;
  constexpr int threads_in_wave = 32;
  const half* grid;
  if constexpr (group_size == 3) {
    grid = (const half*)HIGGS_3_256;
  } else if constexpr (group_size == 5) {
    grid = (const half*)HIGGS_5_256;
  } else if constexpr (group_size == 6) {
    grid = (const half*)HIGGS_6_256;
  }
  

  const int a_gl_stride_8 =  (prob_k / scales_size) * ((scales_size - 1) / group_size + 1);
  const int c_gl_wr = (blockDim.x / threads_in_wave) * blockIdx.x + (threadIdx.x / threads_in_wave);
  const bool pred = c_gl_wr < prob_m;
  int b_gl_rd = 0;
  int a_gl_rd_8 = a_gl_stride_8 * c_gl_wr + (threadIdx.x % threads_in_wave) * (scales_size / threads_in_wave) / group_size;
  const int zeroth_offset = (((threadIdx.x % threads_in_wave) % group_size) * 2) % group_size;

  int a_gl_end_8 = a_gl_stride_8 * (c_gl_wr + 1);

  __shared__ uint4 sh_b[threads_in_wave * (steps_in_wave + 1)];
  float res = 0;
  
  int iters = (prob_k - 1) / scales_size + 1;
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
    if (pred && a_gl_rd_8 < a_gl_end_8) {
      float scale = __half2float(scales[a_gl_rd_8 / ((scales_size - 1) / group_size + 1)]);
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(A) + a_gl_rd_8;

      constexpr int num_values_to_load = ((32 - 1) / group_size + 2) * group_size;
      half dec[num_values_to_load];
      #pragma unroll
      for (int i = 0; i < num_values_to_load / group_size; i++) {
        #pragma unroll
        for (int j = 0; j < group_size; j++) {
          dec[group_size * i + j] = __ldca((grid + group_size * enc[i] + j)); // read 1 halfs at a time
        }
      }

      half* a = reinterpret_cast<half*>(&dec[zeroth_offset]);
      half* b = reinterpret_cast<half*>(&sh_b[b_sh_rd]);
      
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        half res_half = {};
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          res_half = __hfma(a[8 * i + j], b[8 * i + j], res_half);
        }
        iter_res += __half2float(res_half);
      }
      
      iter_res *= scale;
      a_gl_rd_8 += (scales_size - 1) / group_size + 1;
    }
    b_gl_rd += threads_in_wave * steps_in_wave; // Move by scales_size
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

template<int group_size>
void  higgs_Kx256_matvec_cuda(
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

  constexpr int scales_size = 1024;
  int blocks = ceildiv(prob_m, thread_m);
  int threads = 32 * thread_m;
  int shared = 16 * 32 * (scales_size / 32 / 8 + 1);
  cudaFuncSetAttribute(
    HiggsKx256MatVec<group_size>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
  );
  HiggsKx256MatVec<group_size><<<blocks, threads, shared>>>(
    (const uint4*) A,
    (const uint4*) B,
    (uint4*) C,
    (const half*) scales,
    prob_m,
    prob_k
  );
}

template void  higgs_Kx256_matvec_cuda<3>(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ scales,
  int prob_m,
  int prob_k
);

template void  higgs_Kx256_matvec_cuda<5>(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ scales,
  int prob_m,
  int prob_k
);

template void  higgs_Kx256_matvec_cuda<6>(
  const void* __restrict__ A,
  const void* __restrict__ B,
        void* __restrict__ C,
  const void* __restrict__ scales,
  int prob_m,
  int prob_k
);
