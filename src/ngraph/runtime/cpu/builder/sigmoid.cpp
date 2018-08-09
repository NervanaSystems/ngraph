/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

//#include "ngraph/runtime/cpu/kernel/avg_pool.hpp"
#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>
#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"

#include "ngraph/op/sigmoid.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/sigmoid_mul.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Sigmoid)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto input_shape = args[0].get_shape();
                auto out_shape = out[0].get_shape();
                auto input_size = static_cast<int>(shape_size(input_shape));
                auto out_size = static_cast<int>(shape_size(out_shape));

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto input_desc = mkldnn::memory::desc(
                    {input_size},
                    mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                    mkldnn::memory::format::x);
                auto out_desc = mkldnn::memory::desc(
                    {out_size},
                    mkldnn_utils::get_mkldnn_data_type(out[0].get_element_type()),
                    mkldnn::memory::format::x);

                auto sigmoid_index = mkldnn_emitter->build_sigmoid_forward(input_desc, out_desc);

                auto& deps = mkldnn_emitter->get_primitive_deps(sigmoid_index);

                auto functor = [&, sigmoid_index](CPURuntimeContext* ctx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], out_tensor);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, sigmoid_index);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::SigmoidBackprop)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];

                auto input_shape = args[0].get_shape();
                auto delta_shape = args[1].get_shape();
                auto out_shape = out[0].get_shape();
                int input_size = static_cast<int>(shape_size(input_shape));
                int delta_size = static_cast<int>(shape_size(delta_shape));
                int out_size = static_cast<int>(shape_size(out_shape));

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto input_desc = mkldnn::memory::desc(
                    {input_size},
                    mkldnn_utils::get_mkldnn_data_type(args[0].get_element_type()),
                    mkldnn::memory::format::x);
                auto delta_desc = mkldnn::memory::desc(
                    {delta_size},
                    mkldnn_utils::get_mkldnn_data_type(args[1].get_element_type()),
                    mkldnn::memory::format::x);
                auto out_desc = mkldnn::memory::desc(
                    {out_size},
                    mkldnn_utils::get_mkldnn_data_type(out[0].get_element_type()),
                    mkldnn::memory::format::x);

                size_t sigmoid_index =
                    mkldnn_emitter->build_sigmoid_backward(input_desc, delta_desc, out_desc);

                auto& deps = mkldnn_emitter->get_primitive_deps(sigmoid_index);
                auto functor = [&, sigmoid_index](CPURuntimeContext* ctx) {
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg0_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg1_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out_tensor);
                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, sigmoid_index);
                };
                functors.emplace_back(functor);
            }

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

            template <>
            void Builder::BUILDER_DECL(ngraph::op::SigmoidMultiply)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];
                auto tensor_size = shape_size(args[0].get_shape());

                auto sigmoid_mul = static_cast<const ngraph::op::SigmoidMultiply*>(node);
                function<void(CPURuntimeContext*)> functor;

                const size_t index =
                    static_cast<size_t>(sigmoid_mul->get_input_func_type(0)) *
                        static_cast<size_t>(ngraph::op::SigmoidMultiply::FunctionType::NumTypes) +
                    static_cast<size_t>(sigmoid_mul->get_input_func_type(1));

                switch (index)
                {
                case 0 /*Logistic|Logistic*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (in0.exp() * in1.exp()) / ((in0.exp() + 1.f) * (in1.exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 1 /*Logistic|Tanh*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (in0.exp() * ((in1 * 2.f).exp() - 1.f)) /
                                 ((in0.exp() + 1.f) * ((in1 * 2.f).exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 2 /*Logistic|Identity*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (in0.exp() * in1) / (in0.exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 3 /*Tanh|Logistic*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (((in0 * 2.f).exp() - 1.f) * in1.exp()) /
                                 (((in0 * 2.f).exp() + 1.f) * (in1.exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 4 /*Tanh|Tanh*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (((in0 * 2.f).exp() - 1.f) * ((in1 * 2.f).exp() - 1.f)) /
                                 (((in0 * 2.f).exp() + 1.f) * ((in1 * 2.f).exp() + 1.f));
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 5 /*Tanh|Identity*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (((in0 * 2.f).exp() - 1.f) * in1) / ((in0 * 2.f).exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 6 /*Identity|Logistic*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (in0 * in1.exp()) / (in1.exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 7 /*Identity|Tanh*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (in0 * ((in1 * 2.f).exp() - 1.f)) / ((in1 * 2.f).exp() + 1.f);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                case 8 /*Identity|Identity*/:
                    functor = [&, tensor_size](CPURuntimeContext* ctx) {
                        auto in0 = wrap_into_tensor_map<float>(arg0_tensor, tensor_size);
                        auto in1 = wrap_into_tensor_map<float>(arg1_tensor, tensor_size);
                        auto out_tm = wrap_into_tensor_map<float>(out_tensor, tensor_size);
                        auto c = (in0 * in1);
                        out_tm.device(eigen::global_thread_pool_device) = c;
                    };
                    break;
                default: throw ngraph_error("unsupported combination for SigmoidMultiply");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Sigmoid);
            REGISTER_OP_BUILDER(SigmoidBackprop);
            REGISTER_OP_BUILDER(SigmoidMultiply);
        }
    }
}
