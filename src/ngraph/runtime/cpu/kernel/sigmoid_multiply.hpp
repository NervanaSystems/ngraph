//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/cpu_executor.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"

template <typename ElementType>
static Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>>
    wrap_into_tensor_map(void* data, size_t tensor_size)
{
    Eigen::array<Eigen::Index, 1> dims;
    dims[0] = tensor_size;

    Eigen::TensorMap<Eigen::Tensor<ElementType, 1, Eigen::RowMajor>> out(
        static_cast<ElementType*>(data), dims);
    return out;
}

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                using namespace std;
                using namespace ngraph;

                void sigmoid_multiply(void* arg0_tensor,
                                      void* arg1_tensor,
                                      void* out_tensor,
                                      size_t tensor_size,
                                      size_t index,
                                      int arena)
                {
                    auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                    auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                    auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                    switch (index)
                    {
                    case 0 /*Logistic|Logistic*/:
                    {
                        auto c = 1.f / (((-in0).exp() + 1.f) * ((-in1).exp() + 1.f));
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 1 /*Logistic|Tanh*/:
                    {
                        auto c = in1.tanh() / ((-in0).exp() + 1.f);
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 2 /*Logistic|Identity*/:
                    {
                        auto c = in1 / ((-in0).exp() + 1.f);
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 3 /*Tanh|Logistic*/:
                    {
                        auto c = in0.tanh() / ((-in1).exp() + 1.f);
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 4 /*Tanh|Tanh*/:
                    {
                        auto c = in0.tanh() * in1.tanh();
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 5 /*Tanh|Identity*/:
                    {
                        auto c = in0.tanh() * in1;
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 6 /*Identity|Logistic*/:
                    {
                        auto c = in0 / ((-in1).exp() + 1.f);
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 7 /*Identity|Tanh*/:
                    {
                        auto c = in0 * in1.tanh();
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    case 8 /*Identity|Identity*/:
                    {
                        auto c = (in0 * in1);
                        out_tm.device(
                            ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(arena)) = c;
                    }
                    break;
                    default: throw ngraph_error("unsupported combination for SigmoidMultiply");
                    }
                }

                void sigmoid_multiply_backprop(void* arg0_tensor,
                                               void* arg1_tensor,
                                               void* arg2_tensor,
                                               void* out0_tensor,
                                               void* out1_tensor,
                                               size_t tensor_size,
                                               size_t index,
                                               int arena)
                {
                    auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                    auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                    auto delta = wrap_into_tensor_map<float>(arg2_tensor, tensor_size);
                    auto i0_delta = wrap_into_tensor_map<float>(out0_tensor, tensor_size);
                    auto i1_delta = wrap_into_tensor_map<float>(out1_tensor, tensor_size);

                    switch (index)
                    {
                    case 0 /*Logistic|Logistic*/:
                    {
                        auto in0_neg_exp = (-in0).exp();
                        auto in0_log_denominator = in0_neg_exp + 1.f;
                        auto in1_neg_exp = (-in1).exp();
                        auto in1_log_denominator = in1_neg_exp + 1.f;

                        auto i0 = delta * in0_neg_exp /
                                  (in1_log_denominator * in0_log_denominator * in0_log_denominator);
                        auto i1 = delta * in1_neg_exp /
                                  (in0_log_denominator * in1_log_denominator * in1_log_denominator);
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 1 /*Logistic|Tanh*/:
                    {
                        auto in0_neg_exp = (-in0).exp();
                        auto in0_log_denominator = in0_neg_exp + 1.f;
                        auto in1_2exp = (in1 * 2.f).exp();
                        auto in1_tanh_denominator = in1_2exp + 1.f;

                        auto i0 =
                            delta * ((in1_2exp - 1.f) * in0_neg_exp) /
                            (in1_tanh_denominator * in0_log_denominator * in0_log_denominator);
                        auto i1 =
                            delta * (4.f * in1_2exp) /
                            (in0_log_denominator * in1_tanh_denominator * in1_tanh_denominator);
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 2 /*Logistic|Identity*/:
                    {
                        auto in0_neg_exp = (-in0).exp();
                        auto in0_log_denominator = in0_neg_exp + 1.f;

                        auto i0 = delta * (in1 * in0_neg_exp) /
                                  (in0_log_denominator * in0_log_denominator);
                        auto i1 = delta / in0_log_denominator;
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 3 /*Tanh|Logistic*/:
                    {
                        auto in0_2exp = (in0 * 2.f).exp();
                        auto in0_tanh_denominator = in0_2exp + 1.f;
                        auto in1_neg_exp = (-in1).exp();
                        auto in1_log_denominator = in1_neg_exp + 1.f;

                        auto i0 =
                            delta * (4.f * in0_2exp) /
                            (in1_log_denominator * in0_tanh_denominator * in0_tanh_denominator);
                        auto i1 =
                            delta * ((in0_2exp - 1.f) * in1_neg_exp) /
                            (in0_tanh_denominator * in1_log_denominator * in1_log_denominator);
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 4 /*Tanh|Tanh*/:
                    {
                        auto in0_2exp = (in0 * 2.f).exp();
                        auto in0_tanh_denominator = in0_2exp + 1.f;
                        auto in1_2exp = (in1 * 2.f).exp();
                        auto in1_tanh_denominator = in1_2exp + 1.f;

                        auto i0 =
                            delta * (in1_2exp - 1.f) * 4.f * in0_2exp /
                            (in1_tanh_denominator * in0_tanh_denominator * in0_tanh_denominator);
                        auto i1 =
                            delta * (in0_2exp - 1.f) * 4.f * in1_2exp /
                            (in0_tanh_denominator * in1_tanh_denominator * in1_tanh_denominator);
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 5 /*Tanh|Identity*/:
                    {
                        auto in0_2exp = (in0 * 2.f).exp();
                        auto in0_tanh_denominator = in0_2exp + 1.f;

                        auto i0 = delta * in1 * 4.f * in0_2exp /
                                  (in0_tanh_denominator * in0_tanh_denominator);
                        auto i1 = delta * (in0_2exp - 1.f) / in0_tanh_denominator;
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 6 /*Identity|Logistic*/:
                    {
                        auto in1_neg_exp = (-in1).exp();
                        auto in1_log_denominator = in1_neg_exp + 1.f;

                        auto i0 = delta * 1.f / in1_log_denominator;
                        auto i1 =
                            delta * in0 * in1_neg_exp / (in1_log_denominator * in1_log_denominator);
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 7 /*Identity|Tanh*/:
                    {
                        auto in1_2exp = (in1 * 2.f).exp();
                        auto in1_tanh_denominator = in1_2exp + 1.f;

                        auto i0 = delta * (in1_2exp - 1.f) / in1_tanh_denominator;
                        auto i1 = delta * (in0 * (4.f * in1_2exp)) /
                                  (in1_tanh_denominator * in1_tanh_denominator);
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    case 8 /*Identity|Identity*/:
                    {
                        auto i0 = delta * in1;
                        auto i1 = delta * in0;
                        i0_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i0;
                        i1_delta.device(ngraph::runtime::cpu::executor::GetCPUExecutor().get_device(
                            arena)) = i1;
                    }
                    break;
                    default: throw ngraph_error("unsupported combination for SigmoidMultiply");
                    }
                }
            }
        }
    }
}
