//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
                                         const std::vector<TensorWrapper>& args,
                                         const std::vector<TensorWrapper>& out,
                                         bool append_relu,
                                         bool training)
            {
                auto& functors = external_function->get_functors();

                auto arg0_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto arg1_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto arg2_buffer_index = external_function->get_buffer_index(args[2].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());

// Kill clang diagnostics bug
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

                array<size_t, 2> weight_sizes{
                    args[0].get_size() * args[0].get_element_type().size(),
                    args[1].get_size() * args[1].get_element_type().size()};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

                shared_ptr<uint8_t> stacked_weights(new uint8_t[weight_sizes[0] + weight_sizes[1]],
                                                    std::default_delete<uint8_t[]>());

                const float ops_scale = 1.f;
                const float ops_alpha = -0.f; // relu negative slope
                const float ops_beta = 0.f;

                mkldnn::post_ops ops;
                if (append_relu)
                {
                    ops.append_eltwise(
                        ops_scale, mkldnn::algorithm::eltwise_relu, ops_alpha, ops_beta);
                }

                if (training && args.size() == 3)
                {
                    auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());
                    auto out2_buffer_index = external_function->get_buffer_index(out[2].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto batchnorm_desc =
                        mkldnn_emitter->get_batchnorm_forward_desc<OP>(node, true);
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(batchnorm_forward, batchnorm_desc, ops);

                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::FORMAT::nc);

                    // batchnorm forward needs 6 primitives: input, weights, result, mean,
                    // variance, and batch_normalization_forward.
                    auto batchnorm_index = mkldnn_emitter->reserve_primitive_space(6);
                    auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);

                    auto functor = [&,
                                    batchnorm_desc,
                                    weights_desc,
                                    training,
                                    ops,
                                    batchnorm_index,
                                    scratchpad_size,
                                    stacked_weights,
                                    weight_sizes,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    out0_buffer_index,
                                    out1_buffer_index,
                                    out2_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_batchnorm_forward(ctx->mkldnn_memories,
                                                                    ctx->mkldnn_primitives,
                                                                    ctx->mkldnn_scratchpad_mds,
                                                                    batchnorm_desc,
                                                                    weights_desc,
                                                                    training,
                                                                    deps,
                                                                    batchnorm_index,
                                                                    ops);
                        }
                        memcpy(stacked_weights.get(),
                               ctx->buffer_data[arg0_buffer_index],
                               weight_sizes[0]);
                        memcpy(stacked_weights.get() + weight_sizes[0],
                               ctx->buffer_data[arg1_buffer_index],
                               weight_sizes[1]);

                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[1], stacked_weights.get());
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[out0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[out1_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[4], ctx->buffer_data[out2_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            batchnorm_index,
                            deps,
                            cpu::mkldnn_utils::OpType::BATCHNORM3ARGS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto arg4_buffer_index =
                        external_function->get_buffer_index(args[4].get_name());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto batchnorm_desc =
                        mkldnn_emitter->get_batchnorm_forward_desc<OP>(node, false);

                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_2ARGS(batchnorm_forward, batchnorm_desc, ops);

                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::FORMAT::nc);

                    // batchnorm forward needs 6 primitives: input, weights, result, mean,
                    // variance, and batch_normalization_forward.
                    auto batchnorm_index = mkldnn_emitter->reserve_primitive_space(6);
                    auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);

                    auto functor = [&,
                                    batchnorm_desc,
                                    weights_desc,
                                    training,
                                    ops,
                                    batchnorm_index,
                                    scratchpad_size,
                                    stacked_weights,
                                    weight_sizes,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_batchnorm_forward(ctx->mkldnn_memories,
                                                                    ctx->mkldnn_primitives,
                                                                    ctx->mkldnn_scratchpad_mds,
                                                                    batchnorm_desc,
                                                                    weights_desc,
                                                                    training,
                                                                    deps,
                                                                    batchnorm_index,
                                                                    ops);
                        }
                        memcpy(stacked_weights.get(),
                               ctx->buffer_data[arg0_buffer_index],
                               weight_sizes[0]);
                        memcpy(stacked_weights.get() + weight_sizes[0],
                               ctx->buffer_data[arg1_buffer_index],
                               weight_sizes[1]);

                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[0], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg3_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg4_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[3], stacked_weights.get());
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[4], ctx->buffer_data[out0_buffer_index]);

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            batchnorm_index,
                            deps,
                            cpu::mkldnn_utils::OpType::BATCHNORM5ARGS,
                            scratchpad_size);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNormTraining)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const ngraph::op::BatchNormTraining* batchnorm =
                        static_cast<const ngraph::op::BatchNormTraining*>(node);

                    if (args.size() == 3)
                    {
                        auto& functors = external_function->get_functors();

                        std::function<decltype(runtime::cpu::kernel::batch_norm_training<float>)>
                            kernel;

                        SELECT_KERNEL(kernel,
                                      args[0].get_element_type(),
                                      runtime::cpu::kernel::batch_norm_training)

                        auto arg2_shape = args[2].get_shape();
                        auto arg0_buffer_index =
                            external_function->get_buffer_index(args[0].get_name());
                        auto arg1_buffer_index =
                            external_function->get_buffer_index(args[1].get_name());
                        auto arg2_buffer_index =
                            external_function->get_buffer_index(args[2].get_name());

                        auto out0_buffer_index =
                            external_function->get_buffer_index(out[0].get_name());
                        auto out1_buffer_index =
                            external_function->get_buffer_index(out[1].get_name());
                        auto out2_buffer_index =
                            external_function->get_buffer_index(out[2].get_name());
                        auto eps = batchnorm->get_eps_value();

                        auto functor = [&,
                                        kernel,
                                        arg2_shape,
                                        eps,
                                        arg0_buffer_index,
                                        arg1_buffer_index,
                                        arg2_buffer_index,
                                        out0_buffer_index,
                                        out1_buffer_index,
                                        out2_buffer_index](CPURuntimeContext* ctx,
                                                           CPUExecutionContext* /* ectx */) {
                            kernel(eps,
                                   ctx->buffer_data[arg0_buffer_index],
                                   ctx->buffer_data[arg1_buffer_index],
                                   ctx->buffer_data[arg2_buffer_index],
                                   ctx->buffer_data[out0_buffer_index],
                                   ctx->buffer_data[out1_buffer_index],
                                   ctx->buffer_data[out2_buffer_index],
                                   arg2_shape);
                        };
                        functors.emplace_back(functor);
                    }
                    else
                    {
                        auto& functors = external_function->get_functors();

                        std::function<decltype(runtime::cpu::kernel::batch_norm_inference<float>)>
                            kernel;

                        SELECT_KERNEL(kernel,
                                      args[0].get_element_type(),
                                      runtime::cpu::kernel::batch_norm_inference)

                        auto arg2_shape = args[2].get_shape();
                        auto arg0_buffer_index =
                            external_function->get_buffer_index(args[0].get_name());
                        auto arg1_buffer_index =
                            external_function->get_buffer_index(args[1].get_name());
                        auto arg2_buffer_index =
                            external_function->get_buffer_index(args[2].get_name());
                        auto arg3_buffer_index =
                            external_function->get_buffer_index(args[3].get_name());
                        auto arg4_buffer_index =
                            external_function->get_buffer_index(args[4].get_name());

                        auto out0_buffer_index =
                            external_function->get_buffer_index(out[0].get_name());
                        auto eps = batchnorm->get_eps_value();

                        auto functor = [&,
                                        kernel,
                                        arg2_shape,
                                        eps,
                                        arg0_buffer_index,
                                        arg1_buffer_index,
                                        arg2_buffer_index,
                                        arg3_buffer_index,
                                        arg4_buffer_index,
                                        out0_buffer_index](CPURuntimeContext* ctx,
                                                           CPUExecutionContext* /* ectx */) {
                            kernel(eps,
                                   ctx->buffer_data[arg0_buffer_index],
                                   ctx->buffer_data[arg1_buffer_index],
                                   ctx->buffer_data[arg2_buffer_index],
                                   ctx->buffer_data[arg3_buffer_index],
                                   ctx->buffer_data[arg4_buffer_index],
                                   ctx->buffer_data[out0_buffer_index],
                                   arg2_shape);
                        };
                        functors.emplace_back(functor);
                    }
                }
                else
                {
                    build_batch_norm<ngraph::op::BatchNormTraining>(
                        external_function, node, args, out, false, true);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNormInference)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const ngraph::op::BatchNormInference* batchnorm =
                        static_cast<const ngraph::op::BatchNormInference*>(node);

                    auto& functors = external_function->get_functors();

                    std::function<decltype(runtime::cpu::kernel::batch_norm_inference<float>)>
                        kernel;

                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::batch_norm_inference)

                    auto arg2_shape = args[2].get_shape();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto arg4_buffer_index =
                        external_function->get_buffer_index(args[4].get_name());

                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                    auto eps = batchnorm->get_eps_value();

                    auto functor = [&,
                                    kernel,
                                    arg2_shape,
                                    eps,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    out0_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) {
                        kernel(eps,
                               ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[arg2_buffer_index],
                               ctx->buffer_data[arg3_buffer_index],
                               ctx->buffer_data[arg4_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               arg2_shape);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    build_batch_norm<ngraph::op::BatchNormInference>(
                        external_function, node, args, out, false, false);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNormTrainingBackprop)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    const ngraph::op::BatchNormTrainingBackprop* batchnorm =
                        static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);

                    auto& functors = external_function->get_functors();

                    std::function<decltype(runtime::cpu::kernel::batch_norm_backprop<float>)>
                        kernel;

                    SELECT_KERNEL(kernel,
                                  args[0].get_element_type(),
                                  runtime::cpu::kernel::batch_norm_backprop)

                    auto arg2_shape = args[2].get_shape();
                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name()); /* gamma */
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name()); /* beta */
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name()); /* input */
                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name()); /* mean */
                    auto arg4_buffer_index =
                        external_function->get_buffer_index(args[4].get_name()); /* variance */
                    auto arg5_buffer_index =
                        external_function->get_buffer_index(args[5].get_name()); /* delta */

                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                    auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());
                    auto out2_buffer_index = external_function->get_buffer_index(out[2].get_name());
                    auto eps = batchnorm->get_eps_value();

                    auto functor = [&,
                                    kernel,
                                    arg2_shape,
                                    eps,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    arg5_buffer_index,
                                    out0_buffer_index,
                                    out1_buffer_index,
                                    out2_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) {
                        kernel(eps,
                               ctx->buffer_data[arg0_buffer_index],
                               ctx->buffer_data[arg1_buffer_index],
                               ctx->buffer_data[arg2_buffer_index],
                               ctx->buffer_data[arg3_buffer_index],
                               ctx->buffer_data[arg4_buffer_index],
                               ctx->buffer_data[arg5_buffer_index],
                               ctx->buffer_data[out0_buffer_index],
                               ctx->buffer_data[out1_buffer_index],
                               ctx->buffer_data[out2_buffer_index],
                               arg2_shape);
                    };
                    functors.emplace_back(functor);
                }
                else
                {
                    auto& functors = external_function->get_functors();

                    auto arg0_buffer_index =
                        external_function->get_buffer_index(args[0].get_name());
                    auto arg1_buffer_index =
                        external_function->get_buffer_index(args[1].get_name());
                    auto arg2_buffer_index =
                        external_function->get_buffer_index(args[2].get_name());
                    auto arg3_buffer_index =
                        external_function->get_buffer_index(args[3].get_name());
                    auto arg4_buffer_index =
                        external_function->get_buffer_index(args[4].get_name());
                    auto arg5_buffer_index =
                        external_function->get_buffer_index(args[5].get_name());

                    auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                    auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());
                    auto out2_buffer_index = external_function->get_buffer_index(out[2].get_name());

// Kill clang diagnostics bug
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#endif

                    array<size_t, 2> weight_sizes{
                        args[0].get_size() * args[0].get_element_type().size(),
                        args[1].get_size() * args[1].get_element_type().size()};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
                    shared_ptr<uint8_t> stacked_weights(
                        new uint8_t[weight_sizes[0] + weight_sizes[1]],
                        std::default_delete<uint8_t[]>());
                    shared_ptr<uint8_t> stacked_dweights(
                        new uint8_t[weight_sizes[0] + weight_sizes[1]],
                        std::default_delete<uint8_t[]>());

                    auto& mkldnn_emitter = external_function->get_mkldnn_emitter();
                    auto batchnorm_desc = mkldnn_emitter->get_batchnorm_backward_desc(node);
                    auto weights_shape = Shape{2, args[0].get_size()};
                    auto weights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::FORMAT::nc);
                    auto dweights_desc = mkldnn_emitter->build_memory_descriptor(
                        weights_shape, args[0].get_element_type(), mkldnn::memory::FORMAT::nc);
                    auto input_desc = mkldnn_utils::get_input_mkldnn_md(node, 2);

                    // batchnorm backward needs 8 primitives: weights, input, mean, variance,
                    // dinput, dweights, and batch_normalization_backward.
                    auto batchnorm_index = mkldnn_emitter->reserve_primitive_space(8);
                    auto& deps = mkldnn_emitter->get_primitive_deps(batchnorm_index);

                    const ngraph::op::BatchNormTrainingBackprop* batchnorm =
                        static_cast<const ngraph::op::BatchNormTrainingBackprop*>(node);
                    auto eps = batchnorm->get_eps_value();
                    (void)eps; // Use depends on mkl-dnn version
                    size_t scratchpad_size =
                        QUERY_SCRATCHPAD_3ARGS(batchnorm_backward, batchnorm_desc, input_desc, eps);

                    auto functor = [&,
                                    batchnorm_desc,
                                    input_desc,
                                    weights_desc,
                                    dweights_desc,
                                    batchnorm_index,
                                    scratchpad_size,
                                    stacked_weights,
                                    stacked_dweights,
                                    weight_sizes,
                                    arg0_buffer_index,
                                    arg1_buffer_index,
                                    arg2_buffer_index,
                                    arg3_buffer_index,
                                    arg4_buffer_index,
                                    arg5_buffer_index,
                                    out0_buffer_index,
                                    out1_buffer_index,
                                    out2_buffer_index](CPURuntimeContext* ctx,
                                                       CPUExecutionContext* /* ectx */) {
                        if (ctx->first_iteration)
                        {
                            mkldnn_emitter->build_batchnorm_backward(ctx->mkldnn_memories,
                                                                     ctx->mkldnn_primitives,
                                                                     ctx->mkldnn_scratchpad_mds,
                                                                     batchnorm_desc,
                                                                     input_desc,
                                                                     weights_desc,
                                                                     dweights_desc,
                                                                     eps,
                                                                     deps,
                                                                     batchnorm_index);
                        }
                        memcpy(stacked_weights.get(),
                               ctx->buffer_data[arg0_buffer_index],
                               weight_sizes[0]);
                        memcpy(stacked_weights.get() + weight_sizes[0],
                               ctx->buffer_data[arg1_buffer_index],
                               weight_sizes[1]);

                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[0], stacked_weights.get());
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[1], ctx->buffer_data[arg2_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[2], ctx->buffer_data[arg3_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[3], ctx->buffer_data[arg4_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[4], ctx->buffer_data[arg5_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(
                            ctx, deps[5], ctx->buffer_data[out0_buffer_index]);
                        cpu::mkldnn_utils::set_memory_ptr(ctx, deps[6], stacked_dweights.get());

                        cpu::mkldnn_utils::mkldnn_invoke_primitive(
                            ctx,
                            batchnorm_index,
                            deps,
                            cpu::mkldnn_utils::OpType::BATCHNORMBACKPROP,
                            scratchpad_size);

                        memcpy(ctx->buffer_data[out1_buffer_index],
                               stacked_dweights.get(),
                               weight_sizes[0]);
                        memcpy(ctx->buffer_data[out2_buffer_index],
                               stacked_dweights.get() + weight_sizes[0],
                               weight_sizes[1]);
                    };
                    functors.emplace_back(functor);
                }
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNormTrainingRelu)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error("BatchNormRelu is only supported with 4-D MKLDNN kernel.");
                }
                build_batch_norm<ngraph::op::BatchNormTrainingRelu>(
                    external_function, node, args, out, true, true);
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::BatchNormInferenceRelu)
            {
                if (!mkldnn_utils::use_mkldnn_kernel(node))
                {
                    throw ngraph_error("BatchNormRelu is only supported with 4-D MKLDNN kernel.");
                }
                build_batch_norm<ngraph::op::BatchNormInferenceRelu>(
                    external_function, node, args, out, true, false);
            }

            void register_builders_batch_norm_cpp()
            {
                REGISTER_OP_BUILDER(BatchNormTraining);
                REGISTER_OP_BUILDER(BatchNormInference);
                REGISTER_OP_BUILDER(BatchNormTrainingRelu);
                REGISTER_OP_BUILDER(BatchNormInferenceRelu);
                REGISTER_OP_BUILDER(BatchNormTrainingBackprop);
            }
        }
    }
}
