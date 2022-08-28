#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void li_cuda_forward_step_kernel(const scalar_t *__restrict__ i,
                                            scalar_t *__restrict__ v,
                                            scalar_t *__restrict__ leak,
                                            scalar_t *__restrict__ tau_inv,
                                            const float dt, size_t state_size) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < state_size) {
    v[index] = dt * tau_inv[index] * (leak[index] - v[index] + i[index]);
  }
}

torch::Tensor li_cuda_forward_step(torch::Tensor i, torch::Tensor v,
                                   torch::Tensor leak, torch::Tensor tau_inv,
                                   const float dt) {
  const auto batch_size = v.size(0);
  const auto state_size = v.size(1);

  auto new_v = torch::zeros_like(v);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(
      v.type(), "li_forward_step_cuda", ([&] {
        li_cuda_forward_step_kernel<scalar_t><<<blocks, threads>>>(
            i.data<scalar_t>(), v.data<scalar_t>(), leak.data<scalar_t>(),
            tau_inv.data<scalar_t>(), dt, state_size);
      }));

  return new_v;
}

template <typename scalar_t>
__global__ void li_cuda_forward_integral_kernel(
    const scalar_t *__restrict__ i, scalar_t *__restrict__ v,
    scalar_t *__restrict__ leak, scalar_t *__restrict__ tau_inv, const float dt,
    size_t timesteps, size_t state_size) {
  const int index_end = timesteps * blockIdx.x * blockDim.x + threadIdx.x;
  if (index_end < state_size) {
    for (size_t timestep = 0; timestep < timesteps - 1; timestep++) {
      const int neuron_index = blockIdx.x * blockDim.x + threadIdx.x;
      const int index_current =
          timestep * blockIdx.x * blockDim.x + threadIdx.x;
      const int index_next =
          (timestep + 1) * blockIdx.x * blockDim.x + threadIdx.x;
      v[index_next] =
          dt * tau_inv[neuron_index] *
          (leak[neuron_index] - v[index_current] + i[index_current]);
    }
  }
}

torch::Tensor li_cuda_forward_integral(torch::Tensor i, torch::Tensor v,
                                       torch::Tensor leak,
                                       torch::Tensor tau_inv, const float dt) {
  const auto timesteps = v.size(0);
  const auto batch_size = v.size(1);
  const auto state_size = v.size(2);

  auto new_v = torch::zeros_like(i);

  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(
      v.type(), "li_forward_integral_cuda", ([&] {
        li_cuda_forward_integral_kernel<scalar_t><<<blocks, threads>>>(
            i.data<scalar_t>(), v.data<scalar_t>(), leak.data<scalar_t>(),
            tau_inv.data<scalar_t>(), dt, timesteps, state_size);
      }));

  return new_v;
}