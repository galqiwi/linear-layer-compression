#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>


static const __device__ uint32_t HIGGS_2_256[256] = {
  0x3c293e2e,0xb51534e9,0xb0353d77,0x400abcb3,0xbeab345e,0x403e3c93,0x4178b5a8,0xb391ae47,
  0xbe4db5c5,0xbbc9b560,0x3cf3bc38,0x3f54371b,0x2bd9b558,0x3c863fc5,0x3a6eb8a1,0x41b2beea,
  0xc18abdcd,0x3d0bbd9a,0xbeafaad4,0x3e5ebc98,0x35e5315c,0x3f6dba4b,0xbfd73cb3,0x35ac4020,
  0xbd1db58a,0x3b4dbe4f,0x39d83f22,0xbfad3e83,0xaaa2283c,0xbfbf378d,0xbcb83d27,0x39333ae8,
  0xbc96ba3a,0xb2d5b976,0x3b5a3988,0x3d4db25e,0x2ebe3e29,0x3bae3ce7,0xbee9b966,0xc11fc095,
  0x39fc3065,0xb77239eb,0xbcb6bc37,0x394a3ca3,0xb314bd9a,0xbc863596,0xbdd5c03e,0x306aaf71,
  0xb3463978,0xbb2b3c58,0x3afb3baa,0x3dcb3b7c,0x39ccba6c,0xb74fbe63,0x37394121,0xbc923baa,
  0x40623910,0xc06e3a27,0x3fb8b50c,0x3142347c,0xadb6bcad,0x39e74050,0xbc7b2d72,0x36ce3a81,
  0x39bdbc42,0xb3e93e9d,0xb072bede,0x2aa0b9c3,0xb9953afc,0x3416bb68,0xbc103e94,0x31bebcbe,
  0xc27fb9aa,0xc0ad3423,0xa985c027,0xb507c1b5,0x3463b47c,0xac0db821,0xb193b519,0x2fbec100,
  0x35b1bdbe,0x3f162c0d,0x390db62c,0xb76e3daa,0xbb93381c,0xb5c73fff,0xb63bb3ae,0x3ca424ae,
  0x367b3d79,0xb965b847,0x424f3da2,0x3f453ac9,0x3e173569,0x23ba36a1,0xb2013768,0xbff52b5e,
  0x394e38d4,0xba14b507,0x27703f76,0x41483b00,0xb97438d1,0xb8fb3cad,0x3a613667,0xb7883bc6,
  0x3298bf02,0xae8b4199,0xb56fbc5a,0x32383bc5,0xc0dc3d28,0x3db63e78,0xbb6f3a24,0xbb1ab8d7,
  0x3996b016,0x35e835f2,0xbdd9bb37,0x428cba30,0xbb513261,0x367db739,0xba1735ce,0x404bbebe,
  0xc059bd2b,0x3e564052,0x3bb3bce0,0xaa3cb14e,0x3dc12c1d,0xbcde3ffb,0x3948bd75,0xa6cb4090,
  0x3e27390e,0xbec6403f,0xa2703ab9,0xbc78b83f,0x99c038f9,0xbb3ac02f,0x3e8cb82a,0xb6af382f,
  0x4079b92b,0x3e9ec027,0x36b63c4e,0xbb4bbcb4,0xbf203ad4,0x32c539d0,0xbb2dbb1e,0x36413ec3,
  0x3df9ba80,0x3adbb52a,0xb841b5fa,0x428034c0,0xb9842f03,0x3953c10d,0x30b62de3,0xb717c0b1,
  0xb777bd21,0xc11dba21,0xb84240e1,0xbba8bdfe,0xb8fb3ec5,0xb726a7df,0xb3be3b4e,0xbc5e40fe,
  0xba1dbefc,0xbfa4b57c,0xb6ecbae3,0xad02339b,0x37c2b296,0x3d1f3cf8,0xb9d64260,0x3c334101,
  0xbea3be49,0x31193cf8,0xb909b02c,0xbeff41b4,0xbd8730d7,0x3bbabb01,0x3cacc09b,0xc2ac348f,
  0x371dbc95,0x3561aa21,0x3f863eb3,0x4091b195,0xc0e9402b,0x3cf934de,0x31bab818,0x3bd53288,
  0x386bb900,0x3a83bff5,0xbee0c16c,0x2b9abdcb,0x3e65b2dd,0x389e352b,0xbcdc3943,0x385e2a53,
  0xbbd9c123,0xb4fab785,0x3e96be39,0xc23a3d39,0xc17eaf04,0xbd85aef8,0xb994bd85,0x27e8bb9d,
  0xb87135c8,0xbdffbcea,0xb756b902,0x3cd6b9fd,0x3c1f375b,0xb95aba3b,0x41253439,0xb5163cb1,
  0xba504012,0x3cf1bf38,0x37dbbb0b,0x370e3897,0xb42c2ec6,0xbe4438eb,0x3654c02e,0xc095b46d,
  0xbdb53b6b,0xbc72b0a2,0xbce8bd7f,0x37a6428e,0x3c26b860,0x3d50b7b2,0xb7b63144,0x3501b986,
  0x3dec41fa,0xbf44bc45,0xb920bc36,0x403531b8,0xbce4bf01,0xbd97b8e2,0x3b5dabc5,0xbdbd3e76,
  0x408f40b5,0x3c903b52,0x3eb13cfc,0xbe3e3cfb,0x3d38c1e9,0x3cf938ff,0xbb01c2a6,0x3c5cb469,
  0xaad43c5d,0x32ad3817,0x3879becf,0x39923dd1,0xc16a390b,0xb720bfcb,0xb1e9bb2c,0x3605c270,
  0xba873d8f,0xbaf7ac32,0xbd8436b0,0xc036b9eb,0x4055c101,0xc00dbf87,0x4115bc3a,0x40db3e5c,
};


constexpr int HADAMARD_SIZE = 1024;


__global__ void Higgs2x256MatVec(
  const int4* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const half* __restrict__ scales,
  int prob_m,
  int prob_k
) {
  constexpr int group_size = 2;
  int a_gl_stride = prob_k / 8 / 4;
  int a_gl_rd = (blockDim.x / 32) * blockIdx.x + (threadIdx.x / 32);
  int row_number = a_gl_rd;
  bool pred = a_gl_rd < prob_m;
  int b_gl_rd = 0;
  int c_gl_wr = a_gl_rd;
  a_gl_rd = a_gl_stride * a_gl_rd + threadIdx.x % 32;
  int a_gl_end = a_gl_rd + a_gl_stride - threadIdx.x % 32;

  __shared__ int4 sh_b[32 * 5];
  float res = 0;

  int iters = (prob_k - 1) / 1024 + 1;
  while (iters--) {
    // We pad shared memory to avoid bank conflicts during reads
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 4; i += blockDim.x) {
      if (4 * (b_gl_rd + i) < prob_k)
        sh_b[5 * (i / 4) + i % 4] = B[b_gl_rd + i];
    }
    __syncthreads();

    float iter_res = 0;

    int b_sh_rd = 5 * (threadIdx.x % 32);
    if (pred && a_gl_rd < a_gl_end) {
      float scale = __half2float(scales[(a_gl_rd * 32) / HADAMARD_SIZE]);
      const uint8_t* enc = reinterpret_cast<const uint8_t*>(&A[a_gl_rd]);
            
      #pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t dec[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          dec[j] = HIGGS_2_256[enc[4 * i + j]];
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
      a_gl_rd += 32;
    }
    b_gl_rd += 32 * 4;
    res += iter_res;
  }

  if (pred) {
    #pragma unroll
    for (int i = 16; i > 0; i /= 2)
      res += __shfl_down_sync(0xffffffff, res, i);
    if (threadIdx.x % 32 == 0) {
      reinterpret_cast<__half*>(C)[c_gl_wr] = __float2half(res);
    }
  }
}

inline int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

const int THREAD_M = 16;

void  higgs2x256_matvec_cuda(
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
  int shared = 16 * 32 * 9;
  cudaFuncSetAttribute(
    Higgs2x256MatVec, cudaFuncAttributeMaxDynamicSharedMemorySize, shared
  );
  Higgs2x256MatVec<<<blocks, threads, shared>>>(
    (const int4*) A,
    (const int4*) B,
    (int4*) C,
    (const half*) scales,
    prob_m,
    prob_k
  );
}

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
