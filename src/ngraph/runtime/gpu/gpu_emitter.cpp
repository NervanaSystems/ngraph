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
#include <cudnn_v7.h>
#include <iostream>
#include <nvrtc.h>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/allreduce.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concat.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/max.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/min.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/ops/pad.hpp"
#include "ngraph/ops/parameter.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/product.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/reduce_window.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/ops/remainder.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/result.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/select_and_scatter.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/softmax.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/util.hpp"

using namespace std;
namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            void GPU_Emitter::emit_elementwise(
                GPU_ExternalFunction* external_function,
                codegen::CodeWriter& writer,
                const ngraph::Node* node,
                const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
            {
                if (out[0].get_size() == 0)
                {
                    return;
                }

                writer.block_begin("  // " + node->get_name());
                writer << "int count = " << out[0].get_size() << ";\n";
                writer << "if(count == 0) return;\n";
                writer << "ngraph::runtime::gpu::emit_elementwise_op<ngraph::op::"
                       << node->description() << ">(\"" << node->description() << "\""
                       << ", {\"" << args[0].get_type() << "\", \"" << out[0].get_type() << "\"}"
                       << ", count"
                       << ", CUdeviceptr(" << out[0].get_name() << ")";
                for (size_t i = 0; i < args.size(); i++)
                {
                    writer << ", CUdeviceptr(" << args[i].get_name() << ")";
                }
                writer << ");\n";
                writer.block_end();
            }

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
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count);

cudnnOpTensorDescriptor_t opTensorDesc;
cudnnCreateOpTensorDescriptor(&opTensorDesc);
cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_ADD,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN);
    )";

                writer << "cudnnOpTensor(cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << ");\n";
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
                if (arg0_shape.empty() || arg1_shape.empty())
                {
                    auto& first = (arg0_shape.empty() ? args[0] : args[1]);
                    auto& second = (arg0_shape.empty() ? args[1] : args[0]);

                    writer.block_begin("  // " + node->get_name());
                    writer << "int count = " << second.get_size() << ";\n";
                    writer << "cublasScopy("
                           << "cublas_handle,"
                           << "count ," << second.get_name() << ","
                           << "1," << out[0].get_name() << ", 1);\n";
                    writer << "cublasSscal("
                           << "cublas_handle,"
                           << "count ," << first.get_name() << "," << out[0].get_name()
                           << ", 1);\n";
                    writer.block_end();
                    return;
                }

                //set output to 0 if input size is 0
                if (args[0].get_size() == 0 || args[1].get_size() == 0)
                {
                    writer.block_begin("  // " + node->get_name());
                    writer << "runtime::gpu::cuda_memset(" << out[0].get_name() << ", 0, "
                           << out[0].get_size() << " * sizeof(float));\n";
                    writer.block_end();
                    return;
                }

                //case that can be treat as dot1d
                if ((arg0_shape.size() == arg1_shape.size()) &&
                    (arg0_shape.size() == dot->get_reduction_axes_count()))

                {
                    for (int i = 0; i < arg0_shape.size(); i++)
                    {
                        if (arg0_shape[i] != arg1_shape[i])
                        {
                            throw std::runtime_error(
                                "input1 and input2 shape does not match for dot;");
                        }
                    }
                    writer.block_begin("  // " + node->get_name());
                    writer << "cublasSdot("
                           << "cublas_handle," << args[0].get_size() << "," << args[0].get_name()
                           << ","
                           << "1," << args[1].get_name() << ","
                           << "1," << out[0].get_name() << ");\n";
                    writer.block_end();
                }
                else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1) &&
                         (dot->get_reduction_axes_count() == 1))
                {
                    writer.block_begin("  // " + node->get_name());
                    writer << "const float alpha = 1.0;\n";
                    writer << "const float beta  = 0;\n";
                    writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);\n";
                    writer << "cublasSgemv("
                           << "cublas_handle,"
                           << "CUBLAS_OP_T," << arg0_shape[0] << "," << arg0_shape[1] << ","
                           << "&alpha," // Alpha
                           << args[0].get_name() << "," << arg0_shape[1] << ","
                           << args[1].get_name() << ","
                           << "1,"
                           << "&beta," // beta
                           << out[0].get_name() << ","
                           << "1);\n";
                    writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);\n";
                    writer.block_end();
                }
                else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2) &&
                         (dot->get_reduction_axes_count() == 1))
                {
                    // GEMM Call
                    if (arg0_shape[0] != out[0].get_shape()[0] || // m
                        arg1_shape[1] != out[0].get_shape()[1] || // n
                        arg0_shape[1] != arg1_shape[0])           // k
                    {
                        throw std::runtime_error("input and output shape does not match for dot;");
                    }
                    writer.block_begin("  // " + node->get_name());
                    writer << "const float alpha = 1.0;\n";
                    writer << "const float beta  = 0.0;\n";
                    writer << "int m = " << arg0_shape[0] << ";\n";
                    writer << "int n = " << arg1_shape[1] << ";\n";
                    writer << "int k = " << arg0_shape[0] << ";\n";
                    writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);\n";
                    writer << "cublasSgemm("
                           << "cublas_handle,"
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
                           << "n);\n";
                    writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);\n";
                    writer.block_end();
                }
                else
                {
                    throw std::runtime_error(node->get_name() +
                                             " with more then 2D is not implemented.");
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
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count);

cudnnOpTensorDescriptor_t opTensorDesc;
cudnnCreateOpTensorDescriptor(&opTensorDesc);
cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_MAX,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN);
    )";

                writer << "cudnnOpTensor(cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << ");\n";
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
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count);

cudnnOpTensorDescriptor_t opTensorDesc;
cudnnCreateOpTensorDescriptor(&opTensorDesc);
cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_MIN,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN);
    )";

                writer << "cudnnOpTensor(cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << ");\n";
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
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count);

cudnnOpTensorDescriptor_t opTensorDesc;
cudnnCreateOpTensorDescriptor(&opTensorDesc);
cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_ADD,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN);
    )";

                writer << "cudnnOpTensor(cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << ");\n";
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
                //broadcast axes is empty, do a copy
                if (axes.empty())
                {
                    writer.block_begin("  // " + node->get_name());
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                    writer.block_end();
                    return;
                }

                //broadcast axes size is 1, or can be group to 1 (consecutive axes, like 01 or 12 or 123 etc)
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
                    writer << "runtime::gpu::emit_broadcast(\"" << node->description()
                           << "\", CUdeviceptr(" << args[0].get_name() << "), CUdeviceptr("
                           << out[0].get_name() << ")"
                           << ", {\"" << args[0].get_type() << "\", \"" << out[0].get_type()
                           << "\"}"
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
                //  we can just copy.
                if (same_layout || result_shape_product < 2)
                {
                    kernel::emit_memcpyDtD(writer, out[0], args[0]);
                }
                // If there *is* a layout change in the 2D case, we transpose the input.
                else if (arg_rank == 2)
                {
                    // TODO Assert arg0_shape[0] == arg1_shape[0]?
                    writer.indent++;
                    writer << "const float alpha = 1.0;\n";
                    writer << "const float beta = 0;\n";
                    writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);\n";
                    writer << "cublasSgeam("
                           << "cublas_handle,"
                           << "CUBLAS_OP_T,"
                           << "CUBLAS_OP_T," << arg_shape[0] << "," << arg_shape[1] << ","
                           << "&alpha," // Alpha
                           << args[0].get_name() << "," << arg_shape[1] << ","
                           << "&beta," // beta
                           << args[0].get_name() << "," << arg_shape[1] << "," << out[0].get_name()
                           << "," << result_shape[1] << ");\n";
                    writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);\n";
                }
                // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
                else
                {
                    throw runtime_error(
                        "Axis permutation in reshape is not implemented yet for tensors with "
                        "rank>2");
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
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count);

cudnnOpTensorDescriptor_t opTensorDesc;
cudnnCreateOpTensorDescriptor(&opTensorDesc);
cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_MUL,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN);
    )";

                writer << "cudnnOpTensor(cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[1].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << ");\n";
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
                writer << "runtime::gpu::emit_onehot(\"" << node->description()
                       << "\", CUdeviceptr(" << args[0].get_name() << "), CUdeviceptr("
                       << out[0].get_name() << ")"
                       << ", {\"" << args[0].get_type() << "\", \"" << out[0].get_type() << "\"}"
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
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NCHW,
                            /*dataType=*/CUDNN_DATA_FLOAT,
                            /*batch_size=*/1,
                            /*channels=*/1,
                            /*image_height=*/1,
                            /*image_width=*/count);

cudnnOpTensorDescriptor_t opTensorDesc;
cudnnCreateOpTensorDescriptor(&opTensorDesc);
cudnnSetOpTensorDescriptor(opTensorDesc,
                            CUDNN_OP_TENSOR_SQRT,
                            CUDNN_DATA_FLOAT,
                            CUDNN_NOT_PROPAGATE_NAN);
    )";

                writer << "cudnnOpTensor(cudnn_handle,"
                       << "opTensorDesc,"
                       << "&alpha1,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&alpha2,"
                       << "descriptor," << args[0].get_name() << ","
                       << "&beta,"
                       << "descriptor," << out[0].get_name() << ");\n";
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
            void GPU_Emitter::EMITTER_DECL(ngraph::op::Sum)
            {
                auto sum_node = static_cast<const ngraph::op::Sum*>(node);
                auto reduction_axes = sum_node->get_reduction_axes();
                auto& input_shape = args[0].get_shape();
                const std::string input_desc = "input_descriptor";
                const std::string output_desc = "output_descriptor";
                const std::string tensor_type = "CUDNN_DATA_FLOAT";
                const std::string tensor_format = "CUDNN_TENSOR_NCHW";

                writer.block_begin("  // " + node->get_name());
                {
                    if (out[0].get_size() != 0)
                    {
                        // one of args[] axes has zero size, zero output
                        if (args[0].get_size() == 0)
                        {
                            kernel::emit_memset(writer, out[0], 0);
                        }
                        // no change in dimensions, reduction not necessary
                        else if (input_shape.size() == out[0].get_shape().size())
                        {
                            kernel::emit_memcpyDtD(writer, out[0], args[0]);
                        }
                        // descriptors for tensors  with <= 4 dimensions
                        else if (input_shape.size() <= 4)
                        {
                            // construct input tensor descriptor rt impl.
                            std::array<size_t, 4> dimensions;
                            size_t pos = 0;
                            for (size_t i = input_shape.size(); i < 4; i++)
                            {
                                dimensions[pos++] = 1;
                            }
                            for (size_t i = 0; i < input_shape.size(); i++)
                            {
                                dimensions[pos++] = input_shape[i];
                            }

                            kernel::emit_cudnnTensor4dDescriptor(
                                writer, input_desc, tensor_format, tensor_type, dimensions);

                            // mark reduced axes of input tensor for output tensor descriptor
                            for (auto const& idx_dim : reduction_axes)
                            {
                                dimensions[(4 - input_shape.size()) + idx_dim] = 1;
                            }
                            kernel::emit_cudnnTensor4dDescriptor(
                                writer, output_desc, tensor_format, tensor_type, dimensions);

                            // emit sum reduce operation
                            kernel::emit_cudnnReduceTensor(writer,
                                                           args[0],
                                                           out[0],
                                                           "CUDNN_REDUCE_TENSOR_ADD",
                                                           tensor_type,
                                                           "CUDNN_NOT_PROPAGATE_NAN",
                                                           input_desc,
                                                           output_desc,
                                                           1.0,
                                                           0.0);
                        }
                        // descriptors for Nd tensors
                        else
                        {
                            std::vector<size_t> dimensions = input_shape;
                            auto compute_strides = [](const std::vector<size_t>& dim) {
                                std::vector<size_t> strides(dim.size(), 1);
                                std::copy(dim.begin() + 1, dim.end(), strides.begin());
                                for (int64_t i = dim.size() - 2; i >= 0; i--)
                                {
                                    strides[i] *= strides[i + 1];
                                }
                                return strides;
                            };

                            kernel::emit_cudnnTensorNdDescriptor(writer,
                                                                 input_desc,
                                                                 tensor_type,
                                                                 dimensions.size(),
                                                                 dimensions,
                                                                 compute_strides(dimensions));

                            // mark reduced axes of input tensor for output tensor descriptor
                            for (auto const& idx_dim : reduction_axes)
                            {
                                dimensions[idx_dim] = 1;
                            }
                            kernel::emit_cudnnTensorNdDescriptor(writer,
                                                                 output_desc,
                                                                 tensor_type,
                                                                 dimensions.size(),
                                                                 dimensions,
                                                                 compute_strides(dimensions));
                            // emit sum reduce operation
                            kernel::emit_cudnnReduceTensor(writer,
                                                           args[0],
                                                           out[0],
                                                           "CUDNN_REDUCE_TENSOR_ADD",
                                                           tensor_type,
                                                           "CUDNN_NOT_PROPAGATE_NAN",
                                                           input_desc,
                                                           output_desc,
                                                           1.0,
                                                           0.0);
                        }
                    }
                }
                writer.block_end();
                return;
            }
        }
    }
}
