/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <nvrtc.h>
#include <set>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/function_call.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/remainder.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_ops.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/type_info.hpp"
#include "ngraph/util.hpp"

using namespace std;

#define TI(x) type_index(typeid(x))

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Add)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin();
                {
                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();
                    auto index = cudnn_emitter->build_tensor_op(
                        CUDNN_OP_TENSOR_ADD, out[0].get_type(), args[0].get_shape(), 1.0, 1.0, 0);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << ","
                           << args[1].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Convolution)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                auto convolution = static_cast<const ngraph::op::Convolution*>(node);

                size_t conv_index = 0;
                if (convolution->get_padding_below().size() > 3)
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();
                    conv_index = cuda_emitter->build_primitive(convolution);
                }
                else
                {
                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();
                    conv_index = cudnn_emitter->build_primitive(convolution);
                }

                writer << "gpu::invoke_primitive(ctx, " << conv_index << ", ";
                writer << "std::vector<void*>{";
                writer << args[0].get_name() << ", ";
                writer << args[1].get_name() << ", ";
                writer << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data());\n";
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropData)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropData*>(node);

                if (convolution->get_padding_below_forward().size() > 3)
                {
                    throw std::runtime_error(node->get_name() +
                                             "with more than 3D is not implemented.");
                }

                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();
                size_t conv_index = cudnn_emitter->build_primitive(convolution);

                writer << "gpu::invoke_primitive(ctx, " << conv_index << ", ";
                writer << "std::vector<void*>{";
                writer << args[0].get_name() << ", ";
                writer << args[1].get_name() << ", ";
                writer << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data());\n";
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropFilters)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropFilters*>(node);

                if (convolution->get_padding_below_forward().size() > 3)
                {
                    throw std::runtime_error(node->get_name() +
                                             "with more than 3D is not implemented.");
                }

                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();
                size_t conv_index = cudnn_emitter->build_primitive(convolution);

                writer << "gpu::invoke_primitive(ctx, " << conv_index << ", ";
                writer << "std::vector<void*>{";
                writer << args[0].get_name() << ", ";
                writer << args[1].get_name() << ", ";
                writer << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data());\n";
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Dot)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                const ngraph::op::Dot* dot = static_cast<const ngraph::op::Dot*>(node);
                const Shape& arg0_shape = args[0].get_shape();
                const Shape& arg1_shape = args[1].get_shape();
                const Shape& out_shape = out[0].get_shape();
                if (arg0_shape.empty() || arg1_shape.empty())
                {
                    auto& first = (arg0_shape.empty() ? args[0] : args[1]);
                    auto& second = (arg0_shape.empty() ? args[1] : args[0]);

                    writer.block_begin();
                    writer << "int count = " << second.get_size() << ";\n";
                    writer << "CUBLAS_SAFE_CALL(cublasScopy("
                           << "*ctx->cublas_handle,"
                           << "count ," << second.get_name() << ","
                           << "1," << out[0].get_name() << ", 1));\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSscal("
                           << "*ctx->cublas_handle,"
                           << "count ," << first.get_name() << "," << out[0].get_name()
                           << ", 1));\n";
                    writer.block_end();
                    return;
                }

                // set output to 0 if input size is 0
                if (args[0].get_size() == 0 || args[1].get_size() == 0)
                {
                    writer.block_begin();
                    writer << "runtime::gpu::cuda_memset(" << out[0].get_name() << ", 0, "
                           << out[0].get_size() << " * " << out[0].get_element_type().size()
                           << ");\n";
                    writer.block_end();
                    return;
                }

                // case that can be treat as dot1d
                if ((arg0_shape.size() == arg1_shape.size()) &&
                    (arg0_shape.size() == dot->get_reduction_axes_count()))

                {
                    for (int i = 0; i < arg0_shape.size(); i++)
                    {
                        if (arg0_shape[i] != arg1_shape[i])
                        {
                            throw std::invalid_argument(
                                "arg0 and arg1 shape does not match for dot.");
                        }
                    }
                    writer.block_begin();
                    writer << "CUBLAS_SAFE_CALL(cublasSdot("
                           << "*ctx->cublas_handle," << args[0].get_size() << ","
                           << args[0].get_name() << ","
                           << "1," << args[1].get_name() << ","
                           << "1," << out[0].get_name() << "));\n";
                    writer.block_end();
                }
                // matrix vector
                else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) &&
                         (dot->get_reduction_axes_count() == 1))
                {
                    writer.block_begin();
                    writer << "const float alpha = 1.0;\n";
                    writer << "const float beta  = 0;\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSetPointerMode(*ctx->cublas_handle, "
                              "CUBLAS_POINTER_MODE_HOST));\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSgemv("
                           << "*ctx->cublas_handle,"
                           << "CUBLAS_OP_T," << arg0_shape[1] << "," << arg0_shape[0] << ","
                           << "&alpha," // Alpha
                           << args[0].get_name() << "," << arg0_shape[1] << ","
                           << args[1].get_name() << ","
                           << "1,"
                           << "&beta," // beta
                           << out[0].get_name() << ","
                           << "1));\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSetPointerMode(*ctx->cublas_handle, "
                              "CUBLAS_POINTER_MODE_DEVICE));\n";
                    writer.block_end();
                }
                // cases that can be treat as matrix multiply
                else
                {
                    // treat as out[m,n] = arg0[m,k] * arg1[k,n]
                    size_t reduction_axes = dot->get_reduction_axes_count();
                    size_t num_of_axes_for_m = arg0_shape.size() - reduction_axes;
                    size_t num_of_axes_for_n = arg1_shape.size() - reduction_axes;
                    size_t num_of_axes_for_k = reduction_axes;
                    size_t m = 1;
                    size_t n = 1;
                    size_t k = 1;

                    // check if input and output size correct
                    // check and calculate k for arg0 and arg1
                    size_t arg0_k_idx = num_of_axes_for_m; // first axe in arg0 for k
                    size_t arg1_k_idx = 0;                 // first axe in arg1 for k
                    for (size_t i = 0; i < num_of_axes_for_k; i++)
                    {
                        k *= arg0_shape[arg0_k_idx];
                        if (arg0_shape[arg0_k_idx++] != arg1_shape[arg1_k_idx++])
                        {
                            throw std::invalid_argument(
                                "arg0 and arg1 shape does not match for dot.");
                        }
                    }
                    // check and calculate m for arg0 and out
                    size_t arg0_m_idx = 0; // first axe in arg0 for m
                    size_t out_m_idx = 0;  // first axe in out for m
                    for (size_t i = 0; i < num_of_axes_for_m; i++)
                    {
                        m *= arg0_shape[arg0_m_idx];
                        if (arg0_shape[arg0_m_idx++] != out_shape[out_m_idx++])
                        {
                            throw std::invalid_argument(
                                "arg0 and output shape does not match for dot.");
                        }
                    }
                    // check and calculate n for arg1 and out
                    size_t arg1_n_idx = num_of_axes_for_k; // first axe in arg1 for n
                    size_t out_n_idx = num_of_axes_for_m;  // first axe in arg1 for n
                    for (size_t i = 0; i < num_of_axes_for_n; i++)
                    {
                        n *= arg1_shape[arg1_n_idx];
                        if (arg1_shape[arg1_n_idx++] != out_shape[out_n_idx++])
                        {
                            throw std::invalid_argument(
                                "arg1 and output shape does not match for dot.");
                        }
                    }

                    // GEMM Call
                    writer.block_begin();
                    writer << "const float alpha = 1.0;\n";
                    writer << "const float beta  = 0.0;\n";
                    writer << "int m = " << m << ";\n";
                    writer << "int n = " << n << ";\n";
                    writer << "int k = " << k << ";\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSetPointerMode(*ctx->cublas_handle, "
                              "CUBLAS_POINTER_MODE_HOST));\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSgemm("
                           << "*ctx->cublas_handle,"
                           << "CUBLAS_OP_N,"
                           << "CUBLAS_OP_N,"
                           << "n,"
                           << "m,"
                           << "k,"
                           << "&alpha," // Alpha
                           << args[1].get_name() << ","
                           << "n," << args[0].get_name() << ","
                           << "k,"
                           << "&beta," // beta
                           << out[0].get_name() << ","
                           << "n));\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSetPointerMode(*ctx->cublas_handle, "
                              "CUBLAS_POINTER_MODE_DEVICE));\n";
                    writer.block_end();
                }
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Maximum)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin();
                {
                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();
                    auto index = cudnn_emitter->build_tensor_op(
                        CUDNN_OP_TENSOR_MAX, out[0].get_type(), args[0].get_shape(), 1.0, 1.0, 0);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << ","
                           << args[1].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Minimum)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin();
                {
                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();
                    auto index = cudnn_emitter->build_tensor_op(
                        CUDNN_OP_TENSOR_MIN, out[0].get_type(), args[0].get_shape(), 1.0, 1.0, 0);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << ","
                           << args[1].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Broadcast)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                auto broadcast = static_cast<const ngraph::op::Broadcast*>(node);
                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();

                auto& axes = broadcast->get_broadcast_axes();
                // broadcast axes is empty, do a copy
                if (axes.empty())
                {
                    writer.block_begin();
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                    writer.block_end();
                    return;
                }

                auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

                auto bcast_index = cuda_emitter->build_broadcast(
                    {{args[0].get_type(), out[0].get_type()}}, result_shape, axes);
                writer << "gpu::invoke_primitive(ctx, " << bcast_index << ", ";
                writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                writer << ");\n";
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Concat)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                auto concat = static_cast<const ngraph::op::Concat*>(node);
                auto axis = concat->get_concatenation_axis();

                std::vector<std::string> dtypes;
                std::vector<NVShape> input_shapes;
                for (auto arg : args)
                {
                    dtypes.push_back(arg.get_type());
                    input_shapes.push_back(arg.get_shape());
                }
                dtypes.push_back(out[0].get_type());

                writer.block_begin();
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();
                    auto index =
                        cuda_emitter->build_concat(dtypes, input_shapes, axis, out[0].get_shape());

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name();
                    for (size_t i = 1; i < args.size(); i++)
                    {
                        writer << ", " << args[i].get_name();
                    }
                    writer << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Constant)
            {
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Reshape)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                auto reshape = static_cast<const op::Reshape*>(node);
                writer.block_begin();
                auto arg_shape = args[0].get_shape();
                auto arg_rank = arg_shape.size();
                auto result_shape = out[0].get_shape();
                auto input_order = reshape->get_input_order();
                bool same_layout = is_sorted(input_order.begin(), input_order.end());
                size_t result_shape_product = shape_size(result_shape);

                // If there is no layout change or we are just going from 1^n to 1^m or a zero-size tensor,
                // we can just copy.
                if (same_layout || result_shape_product < 2)
                {
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                }
                // If there *is* a layout change in the 2D case, we transpose the input.
                else if (arg_rank == 2)
                {
                    // TODO Assert arg0_shape[0] == arg1_shape[0]?
                    writer << "const float alpha = 1.0;\n";
                    writer << "const float beta = 0;\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSetPointerMode(*ctx->cublas_handle, "
                              "CUBLAS_POINTER_MODE_HOST));\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSgeam("
                           << "*ctx->cublas_handle,"
                           << "CUBLAS_OP_T,"
                           << "CUBLAS_OP_T," << arg_shape[0] << "," << arg_shape[1] << ","
                           << "&alpha," // Alpha
                           << args[0].get_name() << "," << arg_shape[1] << ","
                           << "&beta," // beta
                           << args[0].get_name() << "," << arg_shape[1] << "," << out[0].get_name()
                           << "," << result_shape[1] << "));\n";
                    writer << "CUBLAS_SAFE_CALL(cublasSetPointerMode(*ctx->cublas_handle, "
                              "CUBLAS_POINTER_MODE_DEVICE));\n";
                }
                // Other cases (reordering of axes for tensors with rank>2).
                else
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();
                    auto index = cuda_emitter->build_reshape(
                        {{args[0].get_type(), out[0].get_type()}}, arg_shape, input_order);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::FunctionCall)
            {
                auto function_call = static_cast<const ngraph::op::FunctionCall*>(node);
                shared_ptr<Function> function = function_call->get_functions()[0];

                writer.block_begin();
                {
                    std::vector<string> input_names;
                    std::vector<string> output_names;

                    for (const runtime::gpu::GPU_TensorViewWrapper& input : args)
                    {
                        input_names.push_back(input.get_name());
                    }

                    for (const runtime::gpu::GPU_TensorViewWrapper& output : out)
                    {
                        output_names.push_back(output.get_name());
                    }

                    writer << "void* args[] =\n";
                    writer.block_begin();
                    writer << "\n" << join(input_names, ",\n");
                    writer.block_end();
                    writer << ";\n";

                    writer << "void* out[] =\n";
                    writer.block_begin();
                    writer << "\n" << join(output_names, ",\n");
                    writer.block_end();
                    writer << ";\n";

                    writer << "\n";
                    writer << function->get_name() << "(args, out, ctx);\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Slice)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                auto slice = static_cast<const op::Slice*>(node);

                const auto arg_shape = args[0].get_shape();
                const auto result_shape = out[0].get_shape();
                const Coordinate& lower_bounds = slice->get_lower_bounds();
                const Strides slice_strides = slice->get_strides();

                writer.block_begin();
                if (args[0].get_size() == out[0].get_size())
                {
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                }
                else
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();
                    auto index =
                        cuda_emitter->build_slice({{args[0].get_type(), out[0].get_type()}},
                                                  arg_shape,
                                                  lower_bounds,
                                                  slice_strides,
                                                  result_shape);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Reverse)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                auto reverse = static_cast<const op::Reverse*>(node);

                const auto arg_shape = args[0].get_shape();
                const auto arg_rank = arg_shape.size();
                const auto result_shape = out[0].get_shape();
                const auto reverse_axes = reverse->get_reversed_axes();
                std::vector<uint32_t> reverse_axes_flag(arg_rank, 0);
                for (auto a : reverse_axes)
                {
                    reverse_axes_flag[a] = 1;
                }
                writer.block_begin();
                if (out[0].get_size() == 1)
                {
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                }
                else
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();
                    auto index = cuda_emitter->build_reverse(
                        {{args[0].get_type(), out[0].get_type()}}, arg_shape, reverse_axes_flag);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::ReverseSequence)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                auto rs = static_cast<const ngraph::op::ReverseSequence*>(node);

                size_t bi = rs->get_batch_axis();
                size_t si = rs->get_sequence_axis();
                auto arg_shape0 = args[0].get_shape();
                auto arg_shape1 = args[1].get_shape();
                auto out_shape = out[0].get_shape();

                auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

                auto rs_index = cuda_emitter->build_reverse_sequence(
                    {{args[0].get_type(), args[1].get_type(), out[0].get_type()}},
                    arg_shape0,
                    arg_shape1,
                    out_shape,
                    bi,
                    si);
                writer << "gpu::invoke_primitive(ctx, " << rs_index << ", ";
                writer << "std::vector<void*>{" << args[0].get_name() << ", " << args[1].get_name()
                       << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                writer << ");\n";
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Multiply)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin();
                {
                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();
                    auto index = cudnn_emitter->build_tensor_op(
                        CUDNN_OP_TENSOR_MUL, out[0].get_type(), args[0].get_shape(), 1.0, 1.0, 0);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << ","
                           << args[1].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::OneHot)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                auto onehot = static_cast<const ngraph::op::OneHot*>(node);
                auto arg_shape = args[0].get_shape();
                auto result_shape = out[0].get_shape();
                size_t idx = onehot->get_one_hot_axis();

                writer.block_begin();
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();
                    auto index = cuda_emitter->build_onehot(
                        {{args[0].get_type(), out[0].get_type()}}, arg_shape, result_shape, idx);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Sqrt)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin();
                {
                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();
                    auto index = cudnn_emitter->build_tensor_op(
                        CUDNN_OP_TENSOR_SQRT, out[0].get_type(), args[0].get_shape(), 1.0, 0, 0);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << ","
                           << args[0].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Result)
            {
                writer.block_begin();
                kernel::emit_memcpyDtD(writer, out[0], args[0]);
                writer.block_end();
                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Max)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                const ngraph::op::Max* max = static_cast<const ngraph::op::Max*>(node);
                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();
                auto index = cudnn_emitter->build_primitive(max);

                writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data());\n";

                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Min)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                const ngraph::op::Min* min = static_cast<const ngraph::op::Min*>(node);
                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();
                auto index = cudnn_emitter->build_primitive(min);

                writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data());\n";

                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Sum)
            {
                const ngraph::op::Sum* sum = static_cast<const ngraph::op::Sum*>(node);
                writer.block_begin();
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args[] axes has zero size, zero output
                        if (args[0].get_size() == 0)
                        {
                            kernel::emit_memset(writer, out[0], 0);
                        }
                        else if (args[0].get_size() == out[0].get_size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        else if (out[0].get_shape().size() == 0)
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();
                            auto sum_index =
                                cudnn_emitter->build_reduce_forward(CUDNN_REDUCE_TENSOR_ADD,
                                                                    out[0].get_type(),
                                                                    args[0].get_shape(),
                                                                    sum->get_reduction_axes());

                            writer << "gpu::invoke_primitive(ctx, " << sum_index << ", ";
                            writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                            writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                            writer << ");\n";
                        }
                        else
                        {
                            auto axes_set = sum->get_reduction_axes();
                            ngraph::AxisVector axes_vec;
                            for (auto a : axes_set)
                            {
                                axes_vec.push_back(a);
                            }
                            std::vector<std::string> dtypes;
                            dtypes.push_back(args[0].get_type());
                            dtypes.push_back(out[0].get_type());
                            auto& cuda_emitter =
                                external_function->get_primitive_emitter()->get_cuda_emitter();
                            auto sum_index = cuda_emitter->build_reduce<ngraph::op::Add>(
                                dtypes, args[0].get_shape(), axes_vec);

                            writer << "gpu::invoke_primitive(ctx, " << sum_index << ", ";
                            writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                            writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                            writer << ");\n";
                        }
                    }
                }
                writer.block_end();
                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Product)
            {
                const ngraph::op::Product* product = static_cast<const ngraph::op::Product*>(node);
                writer.block_begin();
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args[] axes has zero size, fill output with 1
                        if (args[0].get_size() == 0)
                        {
                            writer << out[0].get_type() << " init_value = 1;\n";
                            writer << "std::vector<" << out[0].get_type() << "> temp("
                                   << out[0].get_size() << ", init_value);\n";
                            writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                                   << ", (void*)temp.data(), " << out[0].get_size() << " * "
                                   << out[0].get_element_type().size() << ");\n";
                        }
                        else if (args[0].get_size() == out[0].get_size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        // descriptors for tensors  with <= 4 dimensions
                        else
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();
                            auto index =
                                cudnn_emitter->build_reduce_forward(CUDNN_REDUCE_TENSOR_MUL,
                                                                    out[0].get_type(),
                                                                    args[0].get_shape(),
                                                                    product->get_reduction_axes());

                            writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                            writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                            writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                            writer << ");\n";
                        }
                    }
                }
                writer.block_end();
                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Reduce)
            {
                // reduction function supported by GPU
                // CUDNN_REDUCE_TENSOR_ADD
                // CUDNN_REDUCE_TENSOR_MUL
                // CUDNN_REDUCE_TENSOR_MIN
                // CUDNN_REDUCE_TENSOR_MAX
                // CUDNN_REDUCE_TENSOR_AMAX
                // CUDNN_REDUCE_TENSOR_AVG
                // CUDNN_REDUCE_TENSOR_NORM1
                // CUDNN_REDUCE_TENSOR_NORM2
                // CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS

                static const std::unordered_map<std::type_index, cudnnReduceTensorOp_t> reduce_map{
                    {TI(ngraph::op::Add), CUDNN_REDUCE_TENSOR_ADD},
                    {TI(ngraph::op::Multiply), CUDNN_REDUCE_TENSOR_MUL},
                    {TI(ngraph::op::Maximum), CUDNN_REDUCE_TENSOR_MAX},
                    {TI(ngraph::op::Minimum), CUDNN_REDUCE_TENSOR_MIN}};
                const ngraph::op::Reduce* reduce_op = static_cast<const ngraph::op::Reduce*>(node);
                writer.block_begin();
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args0 axes has zero size, zero output, use args1 value
                        if (args[0].get_size() == 0)
                        {
                            writer << out[0].get_type() << " init_value;\n";
                            writer << "runtime::gpu::cuda_memcpyDtH(&init_value, "
                                   << args[1].get_name() << " ,"
                                   << args[1].get_element_type().size() << ");\n";
                            writer << "std::vector<" << out[0].get_type() << "> temp("
                                   << out[0].get_size() << ", init_value);\n";
                            writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                                   << ", (void*)temp.data(), " << out[0].get_size() << " * "
                                   << out[0].get_element_type().size() << ");\n";
                        }
                        else if (args[0].get_size() == out[0].get_size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        else
                        {
                            // in current implementation:
                            // 1. reduction function should only have one op
                            // 2. the op should be in the op_map
                            // otherwise, throw an error message
                            cudnnReduceTensorOp_t reduce_tensor_op;
                            auto reduction_function_ops = reduce_op->get_functions()[0]->get_ops();
                            int op_count = 0;
                            for (auto op : reduction_function_ops)
                            {
                                if (op->is_constant() || op->is_parameter() || op->is_output())
                                {
                                    continue;
                                }
                                op_count++;
                                // Work around a compiler warning (*node inside typeid may have effects
                                // with shared pointers, which is fine here but clang doesn't like it.)
                                auto& fn = *op;
                                auto f_ptr = reduce_map.find(type_index(typeid(fn)));
                                if (f_ptr == reduce_map.end())
                                {
                                    throw std::runtime_error("reduce with function " +
                                                             fn.get_name() +
                                                             " is not implement yet.");
                                }
                                else if (op_count != 1)
                                {
                                    throw std::runtime_error(
                                        "reduce with more than one op is not implement yet.");
                                }
                                else
                                {
                                    reduce_tensor_op = f_ptr->second;
                                }
                            }

                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();
                            auto reduce_index = cudnn_emitter->build_reduce_forward(
                                reduce_tensor_op,
                                out[0].get_type(),
                                args[0].get_shape(),
                                reduce_op->get_reduction_axes());

                            writer << "gpu::invoke_primitive(ctx, " << reduce_index << ", ";
                            writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                            writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                            writer << ");\n";
                        }
                    }
                }
                writer.block_end();
                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::ReduceWindow)
            {
                static const std::unordered_map<std::type_index, ngraph::runtime::gpu::OpName>
                    reduce_window_map{
                        {TI(ngraph::op::Add), ngraph::runtime::gpu::OpName::add},
                        {TI(ngraph::op::Multiply), ngraph::runtime::gpu::OpName::multiply},
                        {TI(ngraph::op::Maximum), ngraph::runtime::gpu::OpName::maximum},
                        {TI(ngraph::op::Minimum), ngraph::runtime::gpu::OpName::minimum}};

                const ngraph::op::ReduceWindow* reduce_window_op =
                    static_cast<const ngraph::op::ReduceWindow*>(node);
                writer.block_begin();
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args0 axes has zero size, zero output, use args1 value
                        if (args[0].get_size() == 0)
                        {
                            writer << out[0].get_type() << " init_value;\n";
                            writer << "runtime::gpu::cuda_memcpyDtH(&init_value, "
                                   << args[1].get_name() << " ,"
                                   << args[1].get_element_type().size() << ");\n";
                            writer << "std::vector<" << out[0].get_type() << "> temp("
                                   << out[0].get_size() << ", init_value);\n";
                            writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                                   << ", (void*)temp.data(), " << out[0].get_size() << " * "
                                   << out[0].get_element_type().size() << ");\n";
                        }
                        else if (args[0].get_size() == out[0].get_size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        else
                        {
                            // in current implementation:
                            // 1. reduction function should only have one op
                            // 2. the op should be in the op_map
                            // otherwise, throw an error message
                            auto reduction_function_ops =
                                reduce_window_op->get_functions()[0]->get_ops();
                            std::unordered_map<std::type_index,
                                               ngraph::runtime::gpu::OpName>::const_iterator it =
                                reduce_window_map.end();
                            int op_count = 0;
                            for (auto op : reduction_function_ops)
                            {
                                if (op->is_constant() || op->is_parameter() || op->is_output())
                                {
                                    continue;
                                }
                                op_count++;
                                // Work around a compiler warning (*node inside typeid may have effects
                                // with shared pointers, which is fine here but clang doesn't like it.)
                                auto& fn = *op;
                                auto f_ptr = reduce_window_map.find(type_index(typeid(fn)));
                                if (op_count != 1)
                                {
                                    throw std::runtime_error(
                                        "reduce with more than one op is not implement yet.");
                                }
                                else if (f_ptr == reduce_window_map.end())
                                {
                                    throw std::runtime_error("reduce with function " +
                                                             fn.get_name() +
                                                             " is not implement yet.");
                                }
                                else
                                {
                                    it = f_ptr;
                                }
                            }

                            if (it == reduce_window_map.end())
                            {
                                throw std::runtime_error(
                                    "no valid op found in reduction function.");
                            }

                            auto& cuda_emitter =
                                external_function->get_primitive_emitter()->get_cuda_emitter();
                            size_t reduce_index;

                            // this dtypes is two build the binary op, expect both input has same type with args[0]
                            std::vector<std::string> dtypes{
                                args[0].get_type(), args[0].get_type(), out[0].get_type()};

                            reduce_index = cuda_emitter->build_reduce_window(
                                it->second,
                                dtypes,
                                args[0].get_shape(),
                                out[0].get_shape(),
                                reduce_window_op->get_window_shape(),
                                reduce_window_op->get_window_movement_strides());

                            writer << "gpu::invoke_primitive(ctx, " << reduce_index << ", ";
                            writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                            writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                            writer << ");\n";
                        }
                    }
                }
                writer.block_end();
                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Pad)
            {
                auto pad = static_cast<const ngraph::op::Pad*>(node);
                writer.block_begin();
                {
                    auto input_shape = args[0].get_shape();
                    auto output_shape = out[0].get_shape();
                    auto padding_below = pad->get_padding_below();
                    auto padding_above = pad->get_padding_above();
                    auto padding_interior = pad->get_padding_interior();

                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();

                    auto pad_index =
                        cuda_emitter->build_pad({{args[0].get_type(), out[0].get_type()}},
                                                input_shape,
                                                output_shape,
                                                padding_below,
                                                padding_above,
                                                padding_interior);
                    writer << "gpu::invoke_primitive(ctx, " << pad_index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << ", "
                           << args[1].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data() ";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::MaxPool)
            {
                // assumes NC{d1,d2,...} format
                auto max_pool = static_cast<const ngraph::op::MaxPool*>(node);

                auto& input_shape = args[0].get_shape();
                auto padding_below = max_pool->get_padding_below();
                auto padding_above = max_pool->get_padding_above();
                if (input_shape.size() < 3)
                {
                    throw std::runtime_error(
                        "MaxPool operation requested for a tensor of less than 3 dimensions. "
                        "Tensors should have at least one spatial dimension, dim(NC{d1...dN}) "
                        "<= 3");
                }
                else if (input_shape.size() > 5)
                {
                    throw std::runtime_error(
                        "Pooling currently only supports up to 3 spatial dimensions.");
                }

                size_t max_pool_index;
                // 1d max pool (NCW)
                if (input_shape.size() == 3)
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();

                    max_pool_index = cuda_emitter->build_primitive(max_pool);
                }
                // 2d and 3d max pool (NCHW)
                else if (input_shape.size() == 4 || input_shape.size() == 5)
                {
                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();

                    max_pool_index = cudnn_emitter->build_primitive(max_pool);
                }
                writer << "gpu::invoke_primitive(ctx, " << max_pool_index << ", ";
                writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                writer << ");\n";
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolBackprop)
            {
                writer.block_begin();
                {
                    auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);
                    auto fp_input_shape = out[0].get_shape();
                    auto fp_output_shape = args[1].get_shape();

                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();

                    if (fp_input_shape.size() >= 4)
                    {
                        auto max_pool_bp_index =
                            cudnn_emitter->build_pooling(CUDNN_POOLING_MAX,
                                                         out[0].get_type(),
                                                         CUDNNEmitter::Prop::Backward,
                                                         fp_input_shape,
                                                         fp_output_shape,
                                                         mpb->get_window_movement_strides(),
                                                         mpb->get_window_shape(),
                                                         mpb->get_padding_below(),
                                                         mpb->get_padding_above());

                        writer << "gpu::invoke_primitive(ctx, " << max_pool_bp_index << ", ";
                        writer << "std::vector<void*>{" << args[0].get_name() << ", "
                               << args[1].get_name() << "}.data(), ";
                        writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                        writer << ");\n";
                    }
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::BatchNorm)
            {
                const ngraph::op::BatchNorm* batchnorm =
                    static_cast<const ngraph::op::BatchNorm*>(node);

                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();

                CUDNNEmitter::Prop direction;
                if (batchnorm->get_training_flag() && args.size() == 3)
                {
                    direction = CUDNNEmitter::Prop::Forward;
                }
                else
                {
                    direction = CUDNNEmitter::Prop::Inference;
                }

                auto bn_index = cudnn_emitter->build_batchnorm(CUDNN_BATCHNORM_SPATIAL,
                                                               out[0].get_type(),
                                                               direction,
                                                               args[2].get_shape(),
                                                               args[0].get_shape(),
                                                               batchnorm->get_eps_value());

                writer.block_begin();
                {
                    writer << "gpu::invoke_primitive(ctx, " << bn_index << ", ";
                    writer << "std::vector<void*>{" << args.front().get_name();
                    for (size_t i = 1; i < args.size(); i++)
                    {
                        writer << ", " << args[i].get_name();
                    }
                    writer << "}.data(), ";
                    writer << "std::vector<void*>{" << out.front().get_name();
                    for (size_t i = 1; i < out.size(); i++)
                    {
                        writer << ", " << out[i].get_name();
                    }
                    writer << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::BatchNormBackprop)
            {
                const ngraph::op::BatchNormBackprop* batchnorm =
                    static_cast<const ngraph::op::BatchNormBackprop*>(node);

                auto& cudnn_emitter =
                    external_function->get_primitive_emitter()->get_cudnn_emitter();

                auto bn_index = cudnn_emitter->build_batchnorm(CUDNN_BATCHNORM_SPATIAL,
                                                               out[0].get_type(),
                                                               CUDNNEmitter::Prop::Backward,
                                                               args[2].get_shape(),
                                                               args[0].get_shape(),
                                                               batchnorm->get_eps_value());

                writer.block_begin();
                {
                    writer << "gpu::invoke_primitive(ctx, " << bn_index << ", ";
                    writer << "std::vector<void*>{" << args.front().get_name();
                    for (size_t i = 1; i < args.size(); i++)
                    {
                        writer << ", " << args[i].get_name();
                    }
                    writer << "}.data(), ";
                    writer << "std::vector<void*>{" << out.front().get_name();
                    for (size_t i = 1; i < out.size(); i++)
                    {
                        writer << ", " << out[i].get_name();
                    }
                    writer << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::GetOutputElement)
            {
                auto get_tuple_element = static_cast<const ngraph::op::GetOutputElement*>(node);

                writer.block_begin();
                writer << "runtime::gpu::cuda_memcpyDtH(" << out[0].get_name() << ", "
                       << args[get_tuple_element->get_n()].get_name() << ", "
                       << out[0].get_size() * out[0].get_element_type().size() << ");\n";
                writer.block_end();
            }

            // assumes NC{d1,d2,d3,...} format
            Shape get_padded_shape(const Shape& input_shape,
                                   const Shape& padding_below,
                                   const Shape& padding_above,
                                   const Shape& padding_interior)
            {
                Shape padded_shape = input_shape;
                int64_t i = input_shape.size() - 1;
                int64_t j = padding_below.size() - 1;
                if (padding_interior.empty())
                {
                    for (; j >= 0; j--, i--)
                    {
                        padded_shape[i] += padding_below[j] + padding_above[j];
                    }
                }
                else
                {
                    for (; j >= 0; j--, i--)
                    {
                        padded_shape[i] = (padded_shape[i] - 1) * padding_interior[j] + 1 +
                                          padding_below[j] + padding_above[j];
                    }
                }
                return padded_shape;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::AvgPool)
            {
                // assumes NC{d1,d2,...} format
                auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);
                writer.block_begin();
                {
                    auto& input_shape = args[0].get_shape();
                    auto& result_shape = out[0].get_shape();
                    auto padding_below = avg_pool->get_padding_below();
                    auto padding_above = avg_pool->get_padding_above();

                    int num_nontrivial_dims = 0;
                    for (int64_t i = input_shape.size() - 1; i > 1; i--)
                    {
                        if (input_shape[i] > 1)
                        {
                            num_nontrivial_dims++;
                        }
                    }

                    size_t avg_pool_index = 0;

                    // if 1d or has asymmetric padding, must handle pooling manually
                    if (input_shape.size() == 3 || padding_below != padding_above)
                    {
                        auto& cuda_emitter =
                            external_function->get_primitive_emitter()->get_cuda_emitter();

                        avg_pool_index =
                            cuda_emitter->build_avg_pool({{args[0].get_type(), out[0].get_type()}},
                                                         input_shape,
                                                         result_shape,
                                                         avg_pool->get_window_shape(),
                                                         avg_pool->get_window_movement_strides(),
                                                         padding_below);
                    }
                    else if (input_shape.size() <= 5)
                    {
                        // 2d and 3d avg pool (NCHW) with either symetric padding or no padding
                        if (input_shape.size() == 4 || input_shape.size() == 5)
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();

                            auto cudnn_avg_type = avg_pool->get_include_padding_in_avg_computation()
                                                      ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                                      : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

                            avg_pool_index = cudnn_emitter->build_pooling(
                                cudnn_avg_type,
                                out[0].get_type(),
                                CUDNNEmitter::Prop::Forward,
                                input_shape,
                                result_shape,
                                avg_pool->get_window_movement_strides(),
                                avg_pool->get_window_shape(),
                                padding_below,
                                padding_above);
                        }
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Pooling currently only supports up to 3 spatial dimensions.");
                    }

                    writer << "gpu::invoke_primitive(ctx, " << avg_pool_index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::AvgPoolBackprop)
            {
                writer.block_begin();
                {
                    auto apb = static_cast<const ngraph::op::AvgPoolBackprop*>(node);
                    auto output_shape = out[0].get_shape();
                    auto delta_shape = args[0].get_shape();

                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();

                    if (output_shape.size() >= 4)
                    {
                        auto cudnn_avg_type = apb->get_include_padding_in_avg_computation()
                                                  ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                                  : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;

                        auto avg_pool_bp_index =
                            cudnn_emitter->build_pooling(cudnn_avg_type,
                                                         out[0].get_type(),
                                                         CUDNNEmitter::Prop::Backward,
                                                         output_shape,
                                                         delta_shape,
                                                         apb->get_window_movement_strides(),
                                                         apb->get_window_shape(),
                                                         apb->get_padding_below(),
                                                         apb->get_padding_above());

                        writer << "gpu::invoke_primitive(ctx, " << avg_pool_bp_index << ", ";
                        // cuDNN backwards pooling requests input and output tensors from
                        // the forward pass but does not use them. It also behaves differently
                        // for max pool vs avg pool. The repetition of args below is to address
                        // this interface in a way that supports both max and avg pooling
                        writer << "std::vector<void*>{" << args[0].get_name() << ", "
                               << args[0].get_name() << "}.data(), ";
                        writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                        writer << ");\n";
                    }
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::ReplaceSlice)
            {
                // assumes NC{d1,d2,...} format
                auto rep_slice = static_cast<const ngraph::op::ReplaceSlice*>(node);
                bool in_place_op = (args[0].get_name() == out[0].get_name());
                writer.block_begin();
                {
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();

                    auto index = cuda_emitter->build_primitive(rep_slice, in_place_op);

                    writer << "gpu::invoke_primitive(ctx, " << index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << ", "
                           << args[1].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Softmax)
            {
                auto softmax = static_cast<const ngraph::op::Softmax*>(node);
                writer.block_begin();
                {
                    size_t softmax_index;
                    if (softmax->get_axes().size() != args[0].get_shape().size())
                    {
                        auto& cuda_emitter =
                            external_function->get_primitive_emitter()->get_cuda_emitter();

                        softmax_index = cuda_emitter->build_primitive(softmax);
                    }
                    else
                    {
                        auto& cudnn_emitter =
                            external_function->get_primitive_emitter()->get_cudnn_emitter();

                        softmax_index = cudnn_emitter->build_softmax(CUDNN_SOFTMAX_FAST,
                                                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                                                     out[0].get_type(),
                                                                     CUDNNEmitter::Prop::Forward,
                                                                     args[0].get_shape());
                    }

                    writer << "gpu::invoke_primitive(ctx, " << softmax_index << ", ";
                    writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                    writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                    writer << ");\n";
                }
                writer.block_end();
            }
        }
    }
}
