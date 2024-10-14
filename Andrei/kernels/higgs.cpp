#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_fp16.h>

#include <fast_hadamard_transform.h>

namespace F = torch::nn::functional;


inline bool check_use_bfloat16(const torch::Tensor& input) {
  auto dtype = input.dtype();
  if (dtype == at::kHalf) {
    return false;
  } else if (dtype == at::kBFloat16) {
    return true;
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "AQLM CUDA kernels only support float16 and bfloat16. Got ",
        dtype.name(),
        ". Please specify the correct `torch_dtype` when loading the model."
      )
    );
  }
}



template<int group_size, int codebook_bits, int hadamard_size>
void  higgs_aligned_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* scales,
  int prob_m,
  int prob_k
);
extern template void  higgs_aligned_matvec_cuda<2, 8, 1024>(const void*, const void*, void*, const void*, int, int);
extern template void  higgs_aligned_matvec_cuda<4, 8, 1024>(const void*, const void*, void*, const void*, int, int);
extern template void  higgs_aligned_matvec_cuda<1, 8, 1024>(const void*, const void*, void*, const void*, int, int);

template<int group_size>
void  higgs_Kx256_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* scales,
  int prob_m,
  int prob_k
);
extern template void higgs_Kx256_matvec_cuda<3>(const void*, const void*, void*, const void*, int, int);
extern template void higgs_Kx256_matvec_cuda<5>(const void*, const void*, void*, const void*, int, int);
extern template void higgs_Kx256_matvec_cuda<6>(const void*, const void*, void*, const void*, int, int);

inline torch::Tensor bias_unflatten_output(
        torch::Tensor& flat_output,
  const std::optional<torch::Tensor>& bias,
  const c10::IntArrayRef& input_sizes
) {
  if (bias.has_value()) {
    flat_output += bias->unsqueeze(0);
  }

  auto output_sizes = input_sizes.vec();
  output_sizes.pop_back();
  output_sizes.push_back(flat_output.size(-1));
  auto output = flat_output.reshape(output_sizes).clone();
  return output;
}

void higgs_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& scales,
  const int group_size,
  const int codebook_bits,
  const bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  if (!use_bfloat16 && codebook_bits == 8) {
    switch (group_size)
    {
    case 1:
      higgs_aligned_matvec_cuda<1, 8, 1024>(A.data_ptr(), B.data_ptr(), C.data_ptr(), scales.data_ptr(), prob_m, prob_k);
      return;
    case 2:
      higgs_aligned_matvec_cuda<2, 8, 1024>(A.data_ptr(), B.data_ptr(), C.data_ptr(), scales.data_ptr(), prob_m, prob_k);
      return;
    case 3:
      higgs_Kx256_matvec_cuda<3>(A.data_ptr(), B.data_ptr(), C.data_ptr(), scales.data_ptr(), prob_m, prob_k);
      return;
    case 4:
      higgs_aligned_matvec_cuda<4, 8, 1024>(A.data_ptr(), B.data_ptr(), C.data_ptr(), scales.data_ptr(), prob_m, prob_k);
      return;
    case 5:
      higgs_Kx256_matvec_cuda<5>(A.data_ptr(), B.data_ptr(), C.data_ptr(), scales.data_ptr(), prob_m, prob_k);
      return;
    case 6:
      higgs_Kx256_matvec_cuda<6>(A.data_ptr(), B.data_ptr(), C.data_ptr(), scales.data_ptr(), prob_m, prob_k);
      return;
    default:
      break;
    }
  }
  throw c10::NotImplementedError(
    {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
    c10::str(
      "FUCK YOU CUNT!"
    )
  );
}

torch::Tensor apply_hadamard(const torch::Tensor& input) {
  auto input_sizes = input.sizes();
  auto flat_input = input.reshape({-1, 1024}).contiguous();
  return fast_hadamard_transform(flat_input, 1.0/32.0).reshape(input_sizes).contiguous();
}


torch::Tensor higgs_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& scales,
  bool do_hadamard,
  const std::optional<torch::Tensor>& bias
) {
  auto had_input = input;
  if (do_hadamard) {
    had_input = apply_hadamard(had_input);
  }

  auto codebook_bits = codes.dtype().itemsize() * 8;
  auto group_size = (input.size(-1) - 1) / codes.size(1) + 1;

  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0);
  auto flat_input = had_input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    // print first 4 elements of input_vec
    // std::cout << "input_vec: " << input_vec[0].item<float>() << " " << input_vec[1].item<float>() << " " << input_vec[2].item<float>() << " " << input_vec[3].item<float>() << std::endl;

    higgs_matvec(
      codes,
      input_vec,
      output_vec,
      scales,
      group_size,
      codebook_bits,
      use_bfloat16
    );
  }
  return bias_unflatten_output(
    flat_output,
    bias,
    input_sizes
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("higgs_matmat", &higgs_matmat, "matrix-matrix product through matvec.");
}
