#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda_fp16.h>


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



void  higgs2x256_matvec_cuda(
  const void* A,
  const void* B,
        void* C,
  const void* scales,
  int prob_m,
  int prob_k
);

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

void higgs2x256_matvec(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& scales,
  const bool use_bfloat16
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(A));
  int prob_m = C.size(0);
  int prob_k = B.size(0);

  if (!use_bfloat16) {
    higgs2x256_matvec_cuda(A.data_ptr(), B.data_ptr(), C.data_ptr(), scales.data_ptr(), prob_m, prob_k);
  } else {
    throw c10::NotImplementedError(
      {__func__, __FILE__, static_cast<uint32_t>(__LINE__)},
      c10::str(
        "FUCK YOU CUNT!"
      )
    );
  }
}

torch::Tensor higgs2x256_matmat(
  const torch::Tensor& input,
  const torch::Tensor& codes,
  const torch::Tensor& scales,
  const std::optional<torch::Tensor>& bias
) {
  bool use_bfloat16 = check_use_bfloat16(input);
  auto input_sizes = input.sizes();
  auto out_features = codes.size(0);
  auto flat_input = input.reshape({-1, input.size(-1)});
  auto flat_output = torch::empty({flat_input.size(0), out_features},
    torch::TensorOptions()
      .dtype(input.dtype())
      .device(input.device())
  );

  for (int i = 0; i < flat_input.size(0); ++i) {
    auto input_vec = flat_input.index({i});
    auto output_vec = flat_output.index({i});
    higgs2x256_matvec(
      codes,
      input_vec,
      output_vec,
      scales,
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
  m.def("higgs2x256_matmat", &higgs2x256_matmat, "2x256 matrix-matrix product through matvec.");
}
