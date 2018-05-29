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
#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_primitive_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_util.hpp"
#include "ngraph/runtime/gpu/type_info.hpp"
#include "ngraph/util.hpp"

using namespace std;

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
#define TI(x) type_index(typeid(x))

static const std::unordered_map<std::type_index, cudnnReduceTensorOp_t> reduce_map{
    {TI(ngraph::op::Add), CUDNN_REDUCE_TENSOR_ADD},
    {TI(ngraph::op::Multiply), CUDNN_REDUCE_TENSOR_MUL},
    {TI(ngraph::op::Maximum), CUDNN_REDUCE_TENSOR_MAX},
    {TI(ngraph::op::Minimum), CUDNN_REDUCE_TENSOR_MIN},
};

// cudnn support elementwised op
// CUDNN_OP_TENSOR_ADD
// CUDNN_OP_TENSOR_MUL
// CUDNN_OP_TENSOR_MIN
// CUDNN_OP_TENSOR_MAX
// CUDNN_OP_TENSOR_SQRT
// CUDNN_OP_TENSOR_NOT

static const std::unordered_map<std::type_index, cudnnOpTensorOp_t> element_op_map{
    {TI(ngraph::op::Add), CUDNN_OP_TENSOR_ADD},
    {TI(ngraph::op::Multiply), CUDNN_OP_TENSOR_MUL},
    {TI(ngraph::op::Maximum), CUDNN_OP_TENSOR_MAX},
    {TI(ngraph::op::Minimum), CUDNN_OP_TENSOR_MIN},
    {TI(ngraph::op::Sqrt), CUDNN_OP_TENSOR_SQRT}};

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
                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
auto& descriptor = descriptors.build<cudnnTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count));

auto& opTensorDesc = descriptors.build<cudnnOpTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_ADD,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN));
    )";

                writer << "CUDNN_SAFE_CALL(cudnnOpTensor(*ctx->cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << "));\n";
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Convolution)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                const std::string args0 = "x_descriptor";
                const std::string args1 = "w_descriptor";
                const std::string out0 = "y_descriptor";
                const std::string conv_descriptor = "conv_descriptor";
                const std::string data_type = "CUDNN_DATA_FLOAT";
                const std::string tensor_format = "CUDNN_TENSOR_NCHW";
                const std::string mode = "CUDNN_CROSS_CORRELATION";
                const std::string conv_algo = "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
                auto convolution = static_cast<const ngraph::op::Convolution*>(node);
                Strides window_dilation_strides = convolution->get_window_dilation_strides();
                Strides window_movement_strides = convolution->get_window_movement_strides();
                Strides data_dilation_strides = convolution->get_data_dilation_strides();
                CoordinateDiff padding = convolution->get_padding_below();
                CoordinateDiff padding_above = convolution->get_padding_above();

                if (padding.size() > 3)
                {
                    throw std::runtime_error(node->get_name() +
                                             "with more than 3D is not implemented.");
                }
                for (auto a : data_dilation_strides)
                {
                    if (a != 1)
                    {
                        throw std::runtime_error(node->get_name() +
                                                 "with data dilation is not implemented.");
                    }
                }
                for (int i = 0; i < padding.size(); i++)
                {
                    if (padding[i] != padding_above[i])
                    {
                        throw std::runtime_error(node->get_name() +
                                                 "with asymmetric padding is not implemented.");
                    }
                }

                writer.block_begin("  // " + node->get_name());
                writer << "float alpha = 1.0;\n";
                writer << "float beta = 0.0;\n";

                // construct input and output tensor descriptor
                kernel::emit_cudnnTensorDescriptor(
                    writer, args0, tensor_format, data_type, args[0].get_shape());
                kernel::emit_cudnnFilterDescriptor(
                    writer, args1, tensor_format, data_type, args[1].get_shape());
                kernel::emit_cudnnTensorDescriptor(
                    writer, out0, tensor_format, data_type, out[0].get_shape());
                kernel::emit_cudnnConvolutionDescriptor(writer,
                                                        conv_descriptor,
                                                        padding,
                                                        window_movement_strides,
                                                        window_dilation_strides,
                                                        mode,
                                                        data_type);

                writer << "size_t workSpaceSizeInBytes = 0;\n";
                writer << "CUDNN_SAFE_CALL(cudnnGetConvolutionForwardWorkspaceSize(*ctx->cudnn_"
                          "handle, "
                       << args0 << ", " << args1 << ", " << conv_descriptor << ", " << out0 << ", "
                       << conv_algo << ", "
                       << "&workSpaceSizeInBytes));\n";

                writer << "void* workspace = "
                          "runtime::gpu::create_gpu_buffer(workSpaceSizeInBytes);\n";
                writer << "CUDNN_SAFE_CALL(cudnnConvolutionForward(*ctx->cudnn_handle, "
                       << "&alpha, " << args0 << ", " << args[0].get_name() << ", " << args1 << ", "
                       << args[1].get_name() << ", " << conv_descriptor << ", " << conv_algo << ", "
                       << "workspace, workSpaceSizeInBytes, "
                       << "&beta, " << out0 << ", " << out[0].get_name() << "));\n";
                writer << "runtime::gpu::free_gpu_buffer(workspace);\n";
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropData)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                const std::string args0 = "w_descriptor";
                const std::string args1 = "dy_descriptor";
                const std::string out0 = "dx_descriptor";
                const std::string conv_descriptor = "conv_descriptor";
                const std::string data_type = "CUDNN_DATA_FLOAT";
                const std::string tensor_format = "CUDNN_TENSOR_NCHW";
                const std::string mode = "CUDNN_CROSS_CORRELATION";
                const std::string conv_algo = "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";

                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropData*>(node);
                Strides window_dilation_strides =
                    convolution->get_window_dilation_strides_forward();
                Strides window_movement_strides =
                    convolution->get_window_movement_strides_forward();
                Strides data_dilation_strides = convolution->get_data_dilation_strides_forward();
                CoordinateDiff padding = convolution->get_padding_below_forward();
                CoordinateDiff padding_above = convolution->get_padding_above_forward();

                if (padding.size() > 3)
                {
                    throw std::runtime_error(node->get_name() +
                                             "with more than 3D is not implemented.");
                }
                for (auto a : data_dilation_strides)
                {
                    if (a != 1)
                    {
                        throw std::runtime_error(node->get_name() +
                                                 "with data dilation is not implemented.");
                    }
                }
                for (int i = 0; i < padding.size(); i++)
                {
                    if (padding[i] != padding_above[i])
                    {
                        throw std::runtime_error(node->get_name() +
                                                 "with asymmetric padding is not implemented.");
                    }
                }

                writer.block_begin("  // " + node->get_name());
                writer << "float alpha = 1.0;\n";
                writer << "float beta = 0.0;\n";

                // construct input and output tensor descriptor
                kernel::emit_cudnnFilterDescriptor(
                    writer, args0, tensor_format, data_type, args[0].get_shape());
                kernel::emit_cudnnTensorDescriptor(
                    writer, args1, tensor_format, data_type, args[1].get_shape());
                kernel::emit_cudnnTensorDescriptor(
                    writer, out0, tensor_format, data_type, out[0].get_shape());
                kernel::emit_cudnnConvolutionDescriptor(writer,
                                                        conv_descriptor,
                                                        padding,
                                                        window_movement_strides,
                                                        window_dilation_strides,
                                                        mode,
                                                        data_type);

                writer << "size_t workSpaceSizeInBytes = 0;\n";
                writer << "CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(*ctx->"
                          "cudnn_handle, "
                       << args0 << ", " << args1 << ", " << conv_descriptor << ", " << out0 << ", "
                       << conv_algo << ", "
                       << "&workSpaceSizeInBytes));\n";

                writer << "void* workspace = "
                          "runtime::gpu::create_gpu_buffer(workSpaceSizeInBytes);\n";

                writer << "CUDNN_SAFE_CALL(cudnnConvolutionBackwardData(*ctx->cudnn_handle, "
                       << "&alpha, " << args0 << ", " << args[0].get_name() << ", " << args1 << ", "
                       << args[1].get_name() << ", " << conv_descriptor << ", " << conv_algo << ", "
                       << "workspace, workSpaceSizeInBytes, "
                       << "&beta, " << out0 << ", " << out[0].get_name() << "));\n";

                writer << "runtime::gpu::free_gpu_buffer(workspace);\n";
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::ConvolutionBackpropFilters)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                const std::string args0 = "x_descriptor";
                const std::string args1 = "dy_descriptor";
                const std::string out0 = "dw_descriptor";
                const std::string conv_descriptor = "conv_descriptor";
                const std::string data_type = "CUDNN_DATA_FLOAT";
                const std::string tensor_format = "CUDNN_TENSOR_NCHW";
                const std::string mode = "CUDNN_CROSS_CORRELATION";
                const std::string conv_algo = "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0";

                auto convolution = static_cast<const ngraph::op::ConvolutionBackpropFilters*>(node);
                Strides window_dilation_strides =
                    convolution->get_window_dilation_strides_forward();
                Strides window_movement_strides =
                    convolution->get_window_movement_strides_forward();
                Strides data_dilation_strides = convolution->get_data_dilation_strides_forward();
                CoordinateDiff padding = convolution->get_padding_below_forward();
                CoordinateDiff padding_above = convolution->get_padding_above_forward();

                if (padding.size() > 3)
                {
                    throw std::runtime_error(node->get_name() +
                                             "with more than 3D is not implemented.");
                }
                for (auto a : data_dilation_strides)
                {
                    if (a != 1)
                    {
                        throw std::runtime_error(node->get_name() +
                                                 "with data dilation is not implemented.");
                    }
                }
                for (int i = 0; i < padding.size(); i++)
                {
                    if (padding[i] != padding_above[i])
                    {
                        throw std::runtime_error(node->get_name() +
                                                 "with asymmetric padding is not implemented.");
                    }
                }

                writer.block_begin("  //data_dilation_ " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer << "float alpha = 1.0;\n";
                writer << "float beta = 0.0;\n";

                // construct input and output tensor descriptor
                kernel::emit_cudnnTensorDescriptor(
                    writer, args0, tensor_format, data_type, args[0].get_shape());
                kernel::emit_cudnnTensorDescriptor(
                    writer, args1, tensor_format, data_type, args[1].get_shape());
                kernel::emit_cudnnFilterDescriptor(
                    writer, out0, tensor_format, data_type, out[0].get_shape());
                kernel::emit_cudnnConvolutionDescriptor(writer,
                                                        conv_descriptor,
                                                        padding,
                                                        window_movement_strides,
                                                        window_dilation_strides,
                                                        mode,
                                                        data_type);

                writer << "size_t workSpaceSizeInBytes = 0;\n";
                writer << "CUDNN_SAFE_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(*ctx->"
                          "cudnn_handle, "
                       << args0 << ", " << args1 << ", " << conv_descriptor << ", " << out0 << ", "
                       << conv_algo << ", "
                       << "&workSpaceSizeInBytes));\n";

                writer << "void* workspace = "
                          "runtime::gpu::create_gpu_buffer(workSpaceSizeInBytes);\n";

                writer << "CUDNN_SAFE_CALL(cudnnConvolutionBackwardFilter(*ctx->cudnn_handle, "
                       << "&alpha, " << args0 << ", " << args[0].get_name() << ", " << args1 << ", "
                       << args[1].get_name() << ", " << conv_descriptor << ", " << conv_algo << ", "
                       << "workspace, workSpaceSizeInBytes, "
                       << "&beta, " << out0 << ", " << out[0].get_name() << "));\n";
                writer << "runtime::gpu::free_gpu_buffer(workspace);\n";
                writer.block_end();
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

                    writer.block_begin("  // " + node->get_name());
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
                    writer.block_begin("  // " + node->get_name());
                    writer << "runtime::gpu::cuda_memset(" << out[0].get_name() << ", 0, "
                           << out[0].get_size() << " * sizeof(float));\n";
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
                    writer.block_begin("  // " + node->get_name());
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
                    writer.block_begin("  // " + node->get_name());
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
                    writer.block_begin("  // " + node->get_name());
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
                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
auto& descriptor = descriptors.build<cudnnTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count));

auto& opTensorDesc = descriptors.build<cudnnOpTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_MAX,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN));
    )";

                writer << "CUDNN_SAFE_CALL(cudnnOpTensor(*ctx->cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << "));\n";
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Minimum)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
auto& descriptor = descriptors.build<cudnnTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count));

auto& opTensorDesc = descriptors.build<cudnnOpTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_MIN,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN));
    )";

                writer << "CUDNN_SAFE_CALL(cudnnOpTensor(*ctx->cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << "));\n";
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Negative)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer += R"(
float alpha1 = -1.0, alpha2 = 0, beta = 0;
auto& descriptor = descriptors.build<cudnnTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count));

auto& opTensorDesc = descriptors.build<cudnnOpTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_ADD,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN));
    )";

                writer << "CUDNN_SAFE_CALL(cudnnOpTensor(*ctx->cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << "));\n";
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
                    writer.block_begin("  // " + node->get_name());
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                    writer.block_end();
                    return;
                }

                // broadcast axes size is 1, or can be group to 1 (consecutive axes, like 01 or 12 or 123 etc)
                vector<int> axes_v;
                std::copy(axes.begin(), axes.end(), std::back_inserter(axes_v));
                std::sort(axes_v.begin(), axes_v.end());
                bool is_one_axes = true;
                if (axes.size() != 1)
                {
                    for (int i = 1; i < axes_v.size(); i++)
                    {
                        if (axes_v[i] != axes_v[i - 1] + 1)
                        {
                            is_one_axes = false;
                            break;
                        }
                    }
                }
                if (is_one_axes)
                {
                    int repeat_times = 1;
                    for (int i = 0; i < axes_v.size(); i++)
                    {
                        repeat_times *= result_shape[axes_v[i]];
                    }

                    int repeat_size = 1;
                    for (int i = *axes_v.rbegin() + 1; i < result_shape.size(); i++)
                    {
                        repeat_size *= result_shape[i];
                    }

                    writer.block_begin("  // " + node->get_name());
                    writer << "runtime::gpu::emit_broadcast(\"" << node->description() << "\", {\""
                           << args[0].get_type() << "\", \"" << out[0].get_type() << "\"}"
                           << ", ctx"
                           << ", CUdeviceptr(" << args[0].get_name() << "), CUdeviceptr("
                           << out[0].get_name() << ")"
                           << ", " << repeat_size << ", " << repeat_times << ", "
                           << out[0].get_size() << ");\n";
                    writer.block_end();
                }
                else
                {
                    throw std::runtime_error(node->get_name() + " is not implemented.");
                }
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

                std::vector<size_t> block_strides(args.size(), 1);
                size_t block_size = 0;
                for (size_t i = 0; i < args.size(); i++)
                {
                    auto arg_shape = args[i].get_shape();
                    auto arg_rank = arg_shape.size();
                    for (size_t j = axis; j < arg_rank; j++)
                    {
                        block_strides[i] *= arg_shape[j];
                    }
                    block_size += block_strides[i];
                }

                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer << "int num_inputs = " << args.size() << ";\n";
                writer << "std::vector<size_t> block_strides_h = {" << join(block_strides)
                       << "};\n";
                writer << "void* block_strides_d = "
                          "runtime::gpu::create_gpu_buffer(sizeof(size_t) * num_inputs);\n";
                writer << "runtime::gpu::cuda_memcpyHtD(block_strides_d, block_strides_h.data(), "
                          "sizeof(size_t) * num_inputs);\n";

                writer << "ngraph::runtime::gpu::emit_concat_op(\"" << node->description() << "\""
                       << ", std::vector<std::string>{";
                for (size_t i = 0; i < args.size(); i++)
                {
                    writer << "\"" << args[i].get_type() << "\", ";
                }
                writer << "\"" << out[0].get_type() << "\"}"
                       << ", ctx"
                       << ", count"
                       << ", " << block_size << ", CUdeviceptr(block_strides_d)"
                       << ", CUdeviceptr(" << out[0].get_name() << ")";
                for (size_t i = 0; i < args.size(); i++)
                {
                    writer << ", CUdeviceptr(" << args[i].get_name() << ")";
                }
                writer << ");\n";
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
                writer.block_begin("  // " + node->get_name());
                auto arg_shape = args[0].get_shape();
                auto arg_rank = arg_shape.size();
                auto result_shape = out[0].get_shape();
                auto input_order = reshape->get_input_order();
                bool same_layout = is_sorted(input_order.begin(), input_order.end());
                size_t result_shape_product = 1;

                for (auto i : result_shape)
                {
                    result_shape_product *= i;
                }
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
                    std::vector<size_t> input_strides(arg_rank);
                    std::vector<size_t> output_strides(arg_rank);
                    std::vector<size_t> trans_strides(arg_rank);
                    size_t stride = 1;
                    for (int i = static_cast<int>(arg_rank) - 1; i >= 0; i--)
                    {
                        input_strides[i] = stride;
                        stride *= arg_shape[i];
                    }
                    stride = 1;
                    for (int i = static_cast<int>(arg_rank) - 1; i >= 0; i--)
                    {
                        output_strides[i] = stride;
                        stride *= arg_shape[input_order[i]];
                    }
                    for (int i = 0; i < arg_rank; i++)
                    {
                        trans_strides[input_order[i]] = output_strides[i];
                    }
                    writer << "size_t rank = " << arg_rank << ";\n";
                    writer << "std::vector<size_t> input_strides_h = {" << input_strides[0] << "UL";
                    for (int i = 1; i < arg_rank; i++)
                    {
                        writer << ", " << input_strides[i] << "UL";
                    }
                    writer << "};\n";

                    writer << "std::vector<size_t> trans_strides_h = {" << trans_strides[0] << "UL";
                    for (int i = 1; i < arg_rank; i++)
                    {
                        writer << ", " << trans_strides[i] << "UL";
                    }
                    writer << "};\n";

                    writer << "void* input_strides_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer << "void* trans_strides_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer
                        << "runtime::gpu::cuda_memcpyHtD(input_strides_d, input_strides_h.data(), "
                           "sizeof(size_t) * rank);\n";
                    writer
                        << "runtime::gpu::cuda_memcpyHtD(trans_strides_d, trans_strides_h.data(), "
                           "sizeof(size_t) * rank);\n";
                    writer << "runtime::gpu::emit_reshape(\"" << node->description() << "\", {\""
                           << args[0].get_type() << "\", \"" << out[0].get_type() << "\"}"
                           << ", ctx"
                           << ", CUdeviceptr(" << args[0].get_name() << "), CUdeviceptr("
                           << out[0].get_name() << ")"
                           << ", "
                           << "CUdeviceptr(input_strides_d), CUdeviceptr(trans_strides_d)"
                           << ", " << arg_rank << ", " << args[0].get_size() << ");\n";
                    writer << "runtime::gpu::free_gpu_buffer(input_strides_d);\n";
                    writer << "runtime::gpu::free_gpu_buffer(trans_strides_d);\n";
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
                const auto arg_rank = arg_shape.size();
                const auto result_shape = out[0].get_shape();
                const Coordinate& lower_bounds = slice->get_lower_bounds();
                const Strides slice_strides = slice->get_strides();
                const auto input_strides = row_major_strides(arg_shape);
                const auto output_strides = row_major_strides(result_shape);

                writer.block_begin("  // " + node->get_name());
                if (args[0].get_size() == out[0].get_size())
                {
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                }
                else
                {
                    writer << "size_t rank = " << arg_rank << ";\n";
                    writer << "std::vector<size_t> input_strides_h = {"
                           << join(input_strides, "UL,") << "UL};\n";
                    writer << "std::vector<size_t> output_strides_h = {"
                           << join(output_strides, "UL,") << "UL};\n";
                    writer << "std::vector<size_t> lower_bounds_h = {" << join(lower_bounds, "UL,")
                           << "UL};\n";
                    writer << "std::vector<size_t> slice_strides_h = {"
                           << join(slice_strides, "UL,") << "UL};\n";

                    writer << "void* input_strides_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer << "void* output_strides_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer << "void* slice_strides_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer << "void* lower_bounds_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer
                        << "runtime::gpu::cuda_memcpyHtD(input_strides_d, input_strides_h.data(), "
                           "sizeof(size_t) * rank);\n";
                    writer << "runtime::gpu::cuda_memcpyHtD(output_strides_d, "
                              "output_strides_h.data(), "
                              "sizeof(size_t) * rank);\n";
                    writer
                        << "runtime::gpu::cuda_memcpyHtD(slice_strides_d, slice_strides_h.data(), "
                           "sizeof(size_t) * rank);\n";
                    writer << "runtime::gpu::cuda_memcpyHtD(lower_bounds_d, lower_bounds_h.data(), "
                              "sizeof(size_t) * rank);\n";

                    writer << "runtime::gpu::emit_slice(\"" << node->description()
                           << "\", CUdeviceptr(" << args[0].get_name() << "), CUdeviceptr("
                           << out[0].get_name() << ")"
                           << ", {\"" << args[0].get_type() << "\", \"" << out[0].get_type()
                           << "\"}"
                           << ", "
                           << "ctx, "
                           << "CUdeviceptr(input_strides_d), CUdeviceptr(lower_bounds_d), "
                              "CUdeviceptr(slice_strides_d), CUdeviceptr(output_strides_d)"
                           << ", " << arg_rank << ", " << out[0].get_size() << ");\n";
                    writer << "runtime::gpu::free_gpu_buffer(input_strides_d);\n";
                    writer << "runtime::gpu::free_gpu_buffer(output_strides_d);\n";
                    writer << "runtime::gpu::free_gpu_buffer(slice_strides_d);\n";
                    writer << "runtime::gpu::free_gpu_buffer(lower_bounds_d);\n";
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
                std::vector<size_t> reverse_axes_flag(arg_rank, 0);
                for (auto a : reverse_axes)
                {
                    reverse_axes_flag[a] = 1;
                }
                writer.block_begin("  // " + node->get_name());
                if (out[0].get_size() == 1)
                {
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                }
                else
                {
                    writer << "size_t rank = " << arg_rank << ";\n";
                    writer << "std::vector<size_t> input_shapes_h = {" << join(arg_shape, "UL,")
                           << "UL};\n";
                    writer << "std::vector<size_t> reverse_axes_h = {"
                           << join(reverse_axes_flag, "UL,") << "UL};\n";

                    writer << "void* input_shapes_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer << "void* reverse_axes_d = "
                              "runtime::gpu::create_gpu_buffer(sizeof(size_t) * rank);\n";
                    writer << "runtime::gpu::cuda_memcpyHtD(input_shapes_d, input_shapes_h.data(), "
                              "sizeof(size_t) * rank);\n";
                    writer << "runtime::gpu::cuda_memcpyHtD(reverse_axes_d, "
                              "reverse_axes_h.data(), "
                              "sizeof(size_t) * rank);\n";

                    writer << "runtime::gpu::emit_reverse(\"" << node->description()
                           << "\", CUdeviceptr(" << args[0].get_name() << "), CUdeviceptr("
                           << out[0].get_name() << ")"
                           << ", {\"" << args[0].get_type() << "\", \"" << out[0].get_type()
                           << "\"}"
                           << ", "
                           << "ctx, "
                           << "CUdeviceptr(input_shapes_d), CUdeviceptr(reverse_axes_d), "
                           << arg_rank << ", " << out[0].get_size() << ");\n";
                    writer << "runtime::gpu::free_gpu_buffer(input_shapes_d);\n";
                    writer << "runtime::gpu::free_gpu_buffer(reverse_axes_d);\n";
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::FunctionCall)
            {
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Multiply)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
auto& descriptor = descriptors.build<cudnnTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count));

auto& opTensorDesc = descriptors.build<cudnnOpTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_MUL,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN));
    )";

                writer << "CUDNN_SAFE_CALL(cudnnOpTensor(*ctx->cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << "));\n";
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
                size_t repeat_times = result_shape[idx];
                size_t repeat_size = 1;
                for (size_t i = idx + 1; i < result_shape.size(); i++)
                {
                    repeat_size *= result_shape[i];
                }

                writer.block_begin("  // " + node->get_name());
                writer << "runtime::gpu::cuda_memset(" << out[0].get_name() << ", 0, "
                       << out[0].get_size() << " * " << out[0].get_element_type().size() << ");\n";
                writer << "runtime::gpu::emit_onehot(\"" << node->description() << "\", {\""
                       << args[0].get_type() << "\", \"" << out[0].get_type() << "\"}"
                       << ", ctx"
                       << ", CUdeviceptr(" << args[0].get_name() << "), CUdeviceptr("
                       << out[0].get_name() << ")"
                       << ", " << repeat_size << ", " << repeat_times << ", " << args[0].get_size()
                       << ");\n";
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Sqrt)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }
                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer += R"(
float alpha1 = 1.0, alpha2 = 0, beta = 0;
auto& descriptor = descriptors.build<cudnnTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count));

auto& opTensorDesc = descriptors.build<cudnnOpTensorDescriptor_t>();
CUDNN_SAFE_CALL(cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_SQRT,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN));
    )";

                writer << "CUDNN_SAFE_CALL(cudnnOpTensor(*ctx->cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << "));\n";
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Result)
            {
                writer.block_begin("  // " + node->get_name());
                kernel::emit_memcpyDtD(writer, out[0], args[0]);
                writer.block_end();
                return;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Max)
            {
                const ngraph::op::Max* max_op = static_cast<const ngraph::op::Max*>(node);
                writer.block_begin("  // " + node->get_name());
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args[] axes has zero size, zero output
                        if (args[0].get_size() == 0)
                        {
                            writer << "std::vector<float> temp(" << out[0].get_size()
                                   << ", -std::numeric_limits<float>::infinity());\n";
                            writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                                   << ", (void*)temp.data(), " << out[0].get_size() << " * "
                                   << out[0].get_element_type().size() << ");\n";
                        }
                        else if (args[0].get_shape().size() == out[0].get_shape().size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        else
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();
                            auto max_index =
                                cudnn_emitter->build_reduce_forward(external_function->ctx().get(),
                                                                    CUDNN_REDUCE_TENSOR_MAX,
                                                                    args[0].get_shape(),
                                                                    max_op->get_reduction_axes());

                            writer << "gpu::invoke_primitive(ctx, " << max_index << ", ";
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
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Min)
            {
                const ngraph::op::Min* min_op = static_cast<const ngraph::op::Min*>(node);
                writer.block_begin("  // " + node->get_name());
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args[] axes has zero size, zero output
                        if (args[0].get_size() == 0)
                        {
                            writer << "std::vector<float> temp(" << out[0].get_size()
                                   << ", std::numeric_limits<float>::infinity());\n";
                            writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                                   << ", (void*)temp.data(), " << out[0].get_size() << " * "
                                   << out[0].get_element_type().size() << ");\n";
                        }
                        else if (args[0].get_shape().size() == out[0].get_shape().size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        else
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();
                            auto min_index =
                                cudnn_emitter->build_reduce_forward(external_function->ctx().get(),
                                                                    CUDNN_REDUCE_TENSOR_MIN,
                                                                    args[0].get_shape(),
                                                                    min_op->get_reduction_axes());

                            writer << "gpu::invoke_primitive(ctx, " << min_index << ", ";
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
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Sum)
            {
                const ngraph::op::Sum* sum = static_cast<const ngraph::op::Sum*>(node);
                writer.block_begin("  // " + node->get_name());
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args[] axes has zero size, zero output
                        if (args[0].get_size() == 0)
                        {
                            kernel::emit_memset(writer, out[0], 0);
                        }
                        else if (args[0].get_shape().size() == out[0].get_shape().size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        // descriptors for tensors  with <= 4 dimensions
                        else
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();
                            auto sum_index =
                                cudnn_emitter->build_reduce_forward(external_function->ctx().get(),
                                                                    CUDNN_REDUCE_TENSOR_ADD,
                                                                    args[0].get_shape(),
                                                                    sum->get_reduction_axes());

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
                writer.block_begin("  // " + node->get_name());
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args[] axes has zero size, fill output with 1
                        if (args[0].get_size() == 0)
                        {
                            writer << "float init_value = 1;\n";
                            writer << "std::vector<float> temp(" << out[0].get_size()
                                   << ", init_value);\n";
                            writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                                   << ", (void*)temp.data(), " << out[0].get_size() << " * "
                                   << out[0].get_element_type().size() << ");\n";
                        }
                        else if (args[0].get_shape().size() == out[0].get_shape().size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        // descriptors for tensors  with <= 4 dimensions
                        else
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();
                            auto index =
                                cudnn_emitter->build_reduce_forward(external_function->ctx().get(),
                                                                    CUDNN_REDUCE_TENSOR_MUL,
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
                const ngraph::op::Reduce* reduce_op = static_cast<const ngraph::op::Reduce*>(node);
                writer.block_begin("  // " + node->get_name());
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args0 axes has zero size, zero output, use args1 value
                        if (args[0].get_size() == 0)
                        {
                            writer << "float init_value;\n";
                            writer << "runtime::gpu::cuda_memcpyDtH(&init_value, "
                                   << args[1].get_name() << " ,"
                                   << args[1].get_element_type().size() << ");\n";
                            writer << "std::vector<float> temp(" << out[0].get_size()
                                   << ", init_value);\n";
                            writer << "runtime::gpu::cuda_memcpyHtD(" << out[0].get_name()
                                   << ", (void*)temp.data(), " << out[0].get_size() << " * "
                                   << out[0].get_element_type().size() << ");\n";
                        }
                        else if (args[0].get_shape().size() == out[0].get_shape().size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        else
                        {
                            // in current implementation:
                            // 1. reduction function should only have one op
                            // 2. the op should be in the op_map
                            // otherwise, throw an error message
                            auto reduction_function_ops = reduce_op->get_functions()[0]->get_ops();
                            cudnnReduceTensorOp_t reduce_tensor_op;
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
                                external_function->ctx().get(),
                                reduce_tensor_op,
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
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Pad)
            {
                auto pad = static_cast<const ngraph::op::Pad*>(node);
                writer.block_begin("  // " + node->get_name());
                {
                    auto input_shape = args[0].get_shape();
                    auto output_shape = out[0].get_shape();
                    auto padding_below = pad->get_padding_below();
                    auto padding_above = pad->get_padding_above();
                    auto padding_interior = pad->get_padding_interior();

                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();

                    auto pad_index =
                        cuda_emitter->build_pad(external_function->ctx().get(),
                                                {{args[0].get_type(), out[0].get_type()}},
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
                writer.block_begin("  // " + node->get_name());
                {
                    auto& input_shape = args[0].get_shape();
                    auto& result_shape = out[0].get_shape();
                    auto padding_below = max_pool->get_padding_below();
                    auto padding_above = max_pool->get_padding_above();
                    if (input_shape.size() < 3)
                    {
                        throw std::runtime_error(
                            "MaxPool operation requested for a tensor of less than 3 dimensions. "
                            "Tensors should have at least one spatial dimension, dim(NC{d1...dN}) "
                            "<= 3");
                    }

                    bool pad_required = false;
                    auto shape_to_pool =
                        get_padded_shape(input_shape, padding_below, padding_above, {});
                    if (shape_to_pool != input_shape)
                    {
                        pad_required = true;
                    }

                    if (pad_required && padding_below != padding_above)
                    {
                        auto& cuda_emitter =
                            external_function->get_primitive_emitter()->get_cuda_emitter();

                        // auto temp_buffer = create_gpu_buffer(shape_size(output_shape)*type_size);
                        auto temp_size =
                            shape_size(shape_to_pool) * args[0].get_element_type().size();
                        writer << "void* pad_buffer = "
                               << "runtime::gpu::create_gpu_buffer(" << temp_size << ");\n";

                        std::stringstream ss;
                        ss << TypeInfo::Get(args[0].get_element_type())->lowest();

                        auto pad_index =
                            cuda_emitter->build_pad(external_function->ctx().get(),
                                                    {{args[0].get_type(), out[0].get_type()}},
                                                    input_shape,
                                                    shape_to_pool,
                                                    padding_below,
                                                    padding_above,
                                                    /*padding_interior*/ {},
                                                    ss.str());

                        writer << "gpu::invoke_primitive(ctx, " << pad_index << ", ";
                        writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                        writer << "std::vector<void*>{pad_buffer}.data()";
                        writer << ");\n";

                        // asymetric padding has been applied, zero out padding vectors to
                        // ensure cudnn does not assume padding during pooling
                        std::fill(padding_below.begin(), padding_below.end(), 0);
                        std::fill(padding_above.begin(), padding_above.end(), 0);
                    }

                    int num_nontrivial_dims = 0;
                    for (int64_t i = shape_to_pool.size() - 1; i > 1; i--)
                    {
                        if (shape_to_pool[i] > 1)
                        {
                            num_nontrivial_dims++;
                        }
                    }

                    if (input_shape.size() <= 5)
                    {
                        size_t max_pool_index = 0;
                        // 1d max pool (NCW)
                        if ((input_shape.size() == 3 || num_nontrivial_dims == 1))
                        {
                            auto& cuda_emitter =
                                external_function->get_primitive_emitter()->get_cuda_emitter();

                            max_pool_index = cuda_emitter->build_1d_max_pool(
                                external_function->ctx().get(),
                                {{args[0].get_type(), out[0].get_type()}},
                                input_shape,
                                result_shape,
                                max_pool->get_window_shape().back(),
                                max_pool->get_window_movement_strides().back());
                        }
                        // 2d and 3d max pool (NCHW)
                        else if (input_shape.size() == 4 || input_shape.size() == 5)
                        {
                            auto& cudnn_emitter =
                                external_function->get_primitive_emitter()->get_cudnn_emitter();

                            max_pool_index = cudnn_emitter->build_pooling(
                                external_function->ctx().get(),
                                CUDNN_POOLING_MAX,
                                CUDNNEmitter::Prop::Forward,
                                shape_to_pool,
                                result_shape,
                                max_pool->get_window_movement_strides(),
                                max_pool->get_window_shape(),
                                padding_below,
                                padding_above);
                        }

                        writer << "gpu::invoke_primitive(ctx, " << max_pool_index << ", ";
                        if (pad_required)
                        {
                            // this would be much cleaner if gpu::memory_primitive's were implemented
                            // and could be bound to callable primitives.
                            writer << "std::vector<void*>{pad_buffer}.data(), ";
                        }
                        else
                        {
                            writer << "std::vector<void*>{" << args[0].get_name() << "}.data(), ";
                        }
                        writer << "std::vector<void*>{" << out[0].get_name() << "}.data()";
                        writer << ");\n";
                    }
                    else
                    {
                        throw std::runtime_error(
                            "Pooling currently only supports up to 3 spatial dimensions.");
                    }

                    if (pad_required)
                    {
                        writer << "runtime::gpu::free_gpu_buffer(pad_buffer);\n";
                    }
                }
                writer.block_end();
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::MaxPoolBackprop)
            {
                writer.block_begin("  // " + node->get_name());
                {
                    auto mpb = static_cast<const ngraph::op::MaxPoolBackprop*>(node);
                    auto fp_input_shape = out[0].get_shape();
                    auto fp_output_shape = args[1].get_shape();

                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();

                    if (fp_input_shape.size() >= 4)
                    {
                        auto max_pool_bp_index =
                            cudnn_emitter->build_pooling(external_function->ctx().get(),
                                                         CUDNN_POOLING_MAX,
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

                auto bn_index = cudnn_emitter->build_batchnorm(external_function->ctx().get(),
                                                               CUDNN_BATCHNORM_SPATIAL,
                                                               direction,
                                                               args[2].get_shape(),
                                                               args[0].get_shape(),
                                                               batchnorm->get_eps_value());

                writer.block_begin("  // " + node->get_name());
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

                auto bn_index = cudnn_emitter->build_batchnorm(external_function->ctx().get(),
                                                               CUDNN_BATCHNORM_SPATIAL,
                                                               CUDNNEmitter::Prop::Backward,
                                                               args[2].get_shape(),
                                                               args[0].get_shape(),
                                                               batchnorm->get_eps_value());

                writer.block_begin("  // " + node->get_name());
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

                writer.block_begin("  // " + node->get_name());
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
                if (padding_interior.size())
                {
                    throw std::runtime_error(
                        "Interior padding support is not yet available on GPU.");
                }

                enum class padtype
                {
                    None,
                    Symmetric,
                    Asymmetric
                };
                auto type = padtype::None;
                for (int i = 0; i < padding_below.size(); i++)
                {
                    if (padding_below[i] != 0 || padding_above[i] != 0)
                    {
                        type = padtype::Symmetric;
                    }
                    if (padding_below[i] != padding_above[i])
                    {
                        type = padtype::Asymmetric;
                        break;
                    }
                }
                if (type == padtype::None)
                {
                    return input_shape;
                }

                Shape padded_shape = input_shape;
                for (int i = 0; i < padding_below.size(); i++)
                {
                    padded_shape[padded_shape.size() - 1 - i] +=
                        (padding_below[padding_below.size() - 1 - i] +
                         padding_above[padding_above.size() - 1 - i]);
                }

                return padded_shape;
            }

            template <>
            void GPU_Emitter::EMITTER_DECL(ngraph::op::AvgPool)
            {
                // assumes NC{d1,d2,...} format
                auto avg_pool = static_cast<const ngraph::op::AvgPool*>(node);
                writer.block_begin("  // " + node->get_name());
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
                    if (input_shape.size() == 3 || num_nontrivial_dims == 1 ||
                        padding_below != padding_above)
                    {
                        auto& cuda_emitter =
                            external_function->get_primitive_emitter()->get_cuda_emitter();

                        avg_pool_index =
                            cuda_emitter->build_avg_pool(external_function->ctx().get(),
                                                         {{args[0].get_type(), out[0].get_type()}},
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
                                external_function->ctx().get(),
                                cudnn_avg_type,
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
                writer.block_begin("  // " + node->get_name());
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
                            cudnn_emitter->build_pooling(external_function->ctx().get(),
                                                         cudnn_avg_type,
                                                         CUDNNEmitter::Prop::Backward,
                                                         output_shape,
                                                         delta_shape,
                                                         apb->get_window_movement_strides(),
                                                         apb->get_window_shape(),
                                                         apb->get_padding_below(),
                                                         apb->get_padding_above());

                        writer << "gpu::invoke_primitive(ctx, " << avg_pool_bp_index << ", ";
                        // CUDNN backwards pooling requests input and output tensors from
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
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Softmax)
            {
                writer.block_begin("  // " + node->get_name());
                {
                    auto softmax = static_cast<const ngraph::op::Softmax*>(node);
                    auto tensor_shape = args[0].get_shape();
                    auto axes = softmax->get_axes();
                    if (axes.size() != tensor_shape.size())
                    {
                        throw std::runtime_error(
                            "Softmax implementation currently only supports all axis activation.");
                    }

                    auto& cudnn_emitter =
                        external_function->get_primitive_emitter()->get_cudnn_emitter();

                    size_t softmax_index =
                        cudnn_emitter->build_softmax(external_function->ctx().get(),
                                                     CUDNN_SOFTMAX_FAST,
                                                     CUDNN_SOFTMAX_MODE_INSTANCE,
                                                     CUDNNEmitter::Prop::Forward,
                                                     tensor_shape);

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
