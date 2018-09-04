//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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
#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"
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
                                      size_t index)
                {
                    auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                    auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                    auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                    switch (index)
                    {
                    case 0 /*Logistic|Logistic*/:
                    {
                        auto c = (in0.exp() * in1.exp()) / ((in0.exp() + 1.f) * (in1.exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 1 /*Logistic|Tanh*/:
                    {
                        auto c = (in0.exp() * ((in1 * 2.f).exp() - 1.f)) /
                                 ((in0.exp() + 1.f) * ((in1 * 2.f).exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 2 /*Logistic|Identity*/:
                    {
                        auto c = (in0.exp() * in1) / (in0.exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 3 /*Tanh|Logistic*/:
                    {
                        auto c = (((in0 * 2.f).exp() - 1.f) * in1.exp()) /
                                 (((in0 * 2.f).exp() + 1.f) * (in1.exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 4 /*Tanh|Tanh*/:
                    {
                        auto c = (((in0 * 2.f).exp() - 1.f) * ((in1 * 2.f).exp() - 1.f)) /
                                 (((in0 * 2.f).exp() + 1.f) * ((in1 * 2.f).exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 5 /*Tanh|Identity*/:
                    {
                        auto c = (((in0 * 2.f).exp() - 1.f) * in1) / ((in0 * 2.f).exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 6 /*Identity|Logistic*/:
                    {
                        auto c = (in0 * in1.exp()) / (in1.exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 7 /*Identity|Tanh*/:
                    {
                        auto c = (in0 * ((in1 * 2.f).exp() - 1.f)) / ((in1 * 2.f).exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    }
                    break;
                    case 8 /*Identity|Identity*/:
                    {
                        auto c = (in0 * in1);
                        out_tm.device(eigen::global_thread_pool_device) = c;
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
                                               size_t index)
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
                        auto i0 = delta * (in1.exp() * in0.exp()) /
                                  ((in1.exp() + 1.f) * ((in0.exp() + 1.f) * (in0.exp() + 1.f)));
                        auto i1 = delta * (in0.exp() * in1.exp()) /
                                  ((in0.exp() + 1.f) * ((in1.exp() + 1.f) * (in1.exp() + 1.f)));
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 1 /*Logistic|Tanh*/:
                    {
                        auto i0 =
                            delta * (((in1 * 2.f).exp() - 1.f) * in0.exp()) /
                            (((in1 * 2.f).exp() + 1.f) * ((in0.exp() + 1.f) * (in0.exp() + 1.f)));
                        auto i1 = delta * (in0.exp() * (4.f * (in1 * 2.f).exp())) /
                                  ((in0.exp() + 1.f) *
                                   (((in1 * 2.f).exp() + 1.f) * ((in1 * 2.f).exp() + 1.f)));
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 2 /*Logistic|Identity*/:
                    {
                        auto i0 =
                            delta * (in1 * in0.exp()) / ((in0.exp() + 1.f) * (in0.exp() + 1.f));
                        auto i1 = delta * in0.exp() / ((in0.exp() + 1.f));
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 3 /*Tanh|Logistic*/:
                    {
                        auto i0 = delta * (in1.exp() * (4.f * (in0 * 2.f).exp())) /
                                  ((in1.exp() + 1.f) * ((in0 * 2.f).exp() + 1.f) *
                                   ((in0 * 2.f).exp() + 1.f));
                        auto i1 =
                            delta * (((in0 * 2.f).exp() - 1.f) * in1.exp()) /
                            (((in0 * 2.f).exp() + 1.f) * ((in1.exp() + 1.f) * (in1.exp() + 1.f)));
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 4 /*Tanh|Tanh*/:
                    {
                        auto i0 = delta * (((in1 * 2.f).exp() - 1.f) * (4.f * (in0 * 2.f).exp())) /
                                  (((in1 * 2.f).exp() + 1.f) *
                                   (((in0 * 2.f).exp() + 1.f) * ((in0 * 2.f).exp() + 1.f)));
                        auto i1 = delta * (((in0 * 2.f).exp() - 1.f) * (4.f * (in1 * 2.f).exp())) /
                                  (((in0 * 2.f).exp() + 1.f) *
                                   (((in1 * 2.f).exp() + 1.f) * ((in1 * 2.f).exp() + 1.f)));
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 5 /*Tanh|Identity*/:
                    {
                        auto i0 = delta * (in1 * (4.f * (in0 * 2.f).exp())) /
                                  (((in0 * 2.f).exp() + 1.f) * ((in0 * 2.f).exp() + 1.f));
                        auto i1 = delta * ((in0 * 2.f).exp() - 1.f) / ((in0 * 2.f).exp() + 1.f);
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 6 /*Identity|Logistic*/:
                    {
                        auto i0 = delta * (in1.exp()) / (in1.exp() + 1.f);
                        auto i1 =
                            delta * (in0 * in1.exp()) / ((in1.exp() + 1.f) * (in1.exp() + 1.f));
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 7 /*Identity|Tanh*/:
                    {
                        auto i0 = delta * ((in1 * 2.f).exp() - 1.f) / ((in1 * 2.f).exp() + 1.f);
                        auto i1 = delta * (in0 * (4.f * (in1 * 2.f).exp())) /
                                  (((in1 * 2.f).exp() + 1.f) * ((in1 * 2.f).exp() + 1.f));
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    case 8 /*Identity|Identity*/:
                    {
                        auto i0 = delta * in1;
                        auto i1 = delta * in0;
                        i0_delta.device(eigen::global_thread_pool_device) = i0;
                        i1_delta.device(eigen::global_thread_pool_device) = i1;
                    }
                    break;
                    default: throw ngraph_error("unsupported combination for SigmoidMultiply");
                    }
                }
            }
        }
    }
}
