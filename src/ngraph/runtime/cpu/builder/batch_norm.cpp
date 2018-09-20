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

#include <array>
#include <cstring>

#include "ngraph/op/batch_norm.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/batchnorm.hpp"
#include "ngraph/runtime/cpu/mkldnn_invoke.hpp"
#include "ngraph/runtime/cpu/mkldnn_utils.hpp"
#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <typename OP>
            static void build_batch_norm(CPU_ExternalFunction* external_function,
                                         const ngraph::Node* node,
                                         const std::vector<TensorViewWrapper>& args,
                                         const std::vector<TensorViewWrapper>& out,
                                         bool append_relu)
            {
                auto& functors = external_function->get_functors();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());

                const OP* batchnorm = static_cast<const OP*>(node);

// Kill clang diagnostics bug
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"

                array<size_t, 2> weight_sizes{
                    args[0].get_size() * args[0].get_element_type().size(),
                    args[1].get_size() * args[1].get_element_type().size()};

#pragma clang diagnostic pop

                shared_ptr<uint8_t> stacked_weights(new uint8_t[weight_sizes[0] + weight_sizes[1]]);

                const float ops_scale = 1.f;
                const float ops_alpha = -0.f; // relu negative slope
                const float ops_beta = 0.f;

                mkldnn::post_ops ops;
                if (append_relu)
                {
                    ops.append_eltwise(
                        ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                }

                if (batchnorm->get_training_flag() && args.size() == 3)
                {
                    auto& out1_tensor = external_function->get_tensor_data(out[1].get_name());
                    auto& out2_tensor = external_function->get_tensor_data(out[2].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto results_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                    auto mean_desc = mkldnn_utils::get_output_mkldnn_md(node, 1);
                    auto variance_desc = mkldnn_utils::get_output_mkldnn_md(node, 2);

                    auto batchnorm_index =
                        mkldnn_emitter->build_batchnorm_forward(input_desc,
                                                                weights_desc,
                                                                results_desc,
                                                                mean_desc,
                                                                variance_desc,
                                                                batchnorm->get_eps_value(),
                                                                false,
                                                                batchnorm->get_training_flag(),
                                                                ops);

                    auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);
                    auto functor = [&, batchnorm_index, stacked_weights, weight_sizes](
                        CPURuntimeContext* ctx) {
                        memcpy(stacked_weights.get(), arg0_tensor, weight_sizes[0]);
                        memcpy(
                            stacked_weights.get() + weight_sizes[0], arg1_tensor, weight_sizes[1]);

                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], stacked_weights.get());
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], out0_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], out1_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[4], out2_tensor);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, batchnorm_index);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                    auto& arg4_tensor = external_function->get_tensor_data(args[4].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                    auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                    auto mean_desc = mkldnn_utils::get_input_mkldnn_md(node, 3);
                    auto variance_desc = mkldnn_utils::get_input_mkldnn_md(node, 4);
                    auto results_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);

                    auto batchnorm_index =
                        mkldnn_emitter->build_batchnorm_forward(input_desc,
                                                                weights_desc,
                                                                results_desc,
                                                                mean_desc,
                                                                variance_desc,
                                                                batchnorm->get_eps_value(),
                                                                true,
                                                                batchnorm->get_training_flag(),
                                                                ops);

                    auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);

                    auto functor = [&, batchnorm_index, stacked_weights, weight_sizes](
                        CPURuntimeContext* ctx) {
                        memcpy(stacked_weights.get(), arg0_tensor, weight_sizes[0]);
                        memcpy(
                            stacked_weights.get() + weight_sizes[0], arg1_tensor, weight_sizes[1]);

                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], arg2_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg3_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg4_tensor);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], stacked_weights.get());
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[4], out0_tensor);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, batchnorm_index);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNorm)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const ngraph::op::BatchNorm* batchnorm =
                        static_cast<const ngraph::op::BatchNorm*>(node);

                    if (batchnorm->get_training_flag() && args.size() == 3)
                    {
                        auto& functors = external_function->get_functors();

                        std::function<decltype(
                            runtime::cpu::kernel::batch_norm_three_outputs<float>)>
                            kernel;

                        SELECT_KERNEL(kernel,
                                      args[0].get_element_type(),
                                      runtime::cpu::kernel::batch_norm_three_outputs);

                        auto arg2_shape = args[2].get_shape();
                        auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                        auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                        auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());

                        auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());
                        auto& out1_tensor = external_function->get_tensor_data(out[1].get_name());
                        auto& out2_tensor = external_function->get_tensor_data(out[2].get_name());
                        auto eps = batchnorm->get_eps_value();

                        auto functor = [&, kernel, arg2_shape, eps](CPURuntimeContext* ctx) {
                            kernel(eps,
                                   arg0_tensor,
                                   arg1_tensor,
                                   arg2_tensor,
                                   out0_tensor,
                                   out1_tensor,
                                   out2_tensor,
                                   arg2_shape);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        auto& functors = external_function->get_functors();

                        std::function<decltype(runtime::cpu::kernel::batch_norm_one_output<float>)>
                            kernel;

                        SELECT_KERNEL(kernel,
                                      args[0].get_element_type(),
                                      runtime::cpu::kernel::batch_norm_one_output);

                        auto arg2_shape = args[2].get_shape();
                        auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                        auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                        auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                        auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                        auto& arg4_tensor = external_function->get_tensor_data(args[4].get_name());

                        auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());
                        auto eps = batchnorm->get_eps_value();

                        auto functor = [&, kernel, arg2_shape, eps](CPURuntimeContext* ctx) {
                            kernel(eps,
                                   arg0_tensor,
                                   arg1_tensor,
                                   arg2_tensor,
                                   arg3_tensor,
                                   arg4_tensor,
                                   out0_tensor,
                                   arg2_shape);
                        };
                        functors.emplace_back(functor);
                    }
                }
                else
                {
                    build_batch_norm<ngraph::op::BatchNorm>(
                        external_function, node, args, out, false);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNormBackprop)
            {
                const ngraph::op::BatchNormBackprop* batchnorm =
                    static_cast<const ngraph::op::BatchNormBackprop*>(node);

                auto& functors = external_function->get_functors();

                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());
                auto& arg3_tensor = external_function->get_tensor_data(args[3].get_name());
                auto& arg4_tensor = external_function->get_tensor_data(args[4].get_name());
                auto& arg5_tensor = external_function->get_tensor_data(args[5].get_name());

                auto& out0_tensor = external_function->get_tensor_data(out[0].get_name());
                auto& out1_tensor = external_function->get_tensor_data(out[1].get_name());
                auto& out2_tensor = external_function->get_tensor_data(out[2].get_name());

// Kill clang diagnostics bug
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"

                array<size_t, 2> weight_sizes{
                    args[0].get_size() * args[0].get_element_type().size(),
                    args[1].get_size() * args[1].get_element_type().size()};

#pragma clang diagnostic pop
                shared_ptr<uint8_t> stacked_weights(new uint8_t[weight_sizes[0] + weight_sizes[1]]);
                shared_ptr<uint8_t> stacked_dweights(
                    new uint8_t[weight_sizes[0] + weight_sizes[1]]);

                auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                auto weights_shape = Shape{2, args[0].get_size()};
                auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                    weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);
                auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);
                auto mean_desc = mkldnn_utils::get_input_mkldnn_md(node, 3);
                auto variance_desc = mkldnn_utils::get_input_mkldnn_md(node, 4);
                auto delta_desc = mkldnn_utils::get_input_mkldnn_md(node, 5);
                auto dinput_desc = mkldnn_utils::get_output_mkldnn_md(node, 0);
                auto dweights_desc = mkldnn_emitter->build_memory_descriptor(
                    weights_shape, args[0].get_element_type(), mkldnn::memory::format::nc);

                auto batchnorm_index =
                    mkldnn_emitter->build_batchnorm_backward(weights_desc,
                                                             input_desc,
                                                             mean_desc,
                                                             variance_desc,
                                                             delta_desc,
                                                             dinput_desc,
                                                             dweights_desc,
                                                             batchnorm->get_eps_value());

                auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);

                auto functor = [&,
                                batchnorm_index,
                                stacked_weights,
                                stacked_dweights,
                                weight_sizes](CPURuntimeContext* ctx) {
                    memcpy(stacked_weights.get(), arg0_tensor, weight_sizes[0]);
                    memcpy(stacked_weights.get() + weight_sizes[0], arg1_tensor, weight_sizes[1]);

                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], stacked_weights.get());
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], arg2_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[2], arg3_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], arg4_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[4], arg5_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[5], out0_tensor);
                    cpu::mkldnn_utils::set_memory_ptr(ctx, deps[6], stacked_dweights.get());

                    cpu::mkldnn_utils::mkldnn_invoke_primitive(ctx, batchnorm_index);

                    memcpy(out1_tensor, stacked_dweights.get(), weight_sizes[0]);
                    memcpy(out2_tensor, stacked_dweights.get() + weight_sizes[0], weight_sizes[1]);
                };
                functors.emplace_back(functor);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNormRelu)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error("BatchNormRelu is only supported with 4-D MKLDNN kernel.");
                }
                build_batch_norm<ngraph::op::BatchNormRelu>(
                    external_function, node, args, out, true);
            }
            REGISTER_OP_BUILDER(BatchNorm);
            REGISTER_OP_BUILDER(BatchNormRelu);
            REGISTER_OP_BUILDER(BatchNormBackprop);
        }
    }
}
