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
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/concat.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/runtime/gpu/gpu_cuda_kernel_emitters.hpp"
#include "ngraph/runtime/gpu/gpu_emitter.hpp"
#include "ngraph/runtime/gpu/gpu_kernel_emitters.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

void runtime::gpu::GPU_Emitter::EmitNop(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitUnaryElementwise(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    writer << "{  // " << n->get_name() << "\n";
    writer.indent++;
    writer << "int count = " << out[0].get_size() << ";\n";
    writer << "ngraph::runtime::gpu::emit_unary_elementwise_op<ngraph::op::" << n->description()
           << ">((void*) " << args[0].get_name() << ", (void*) " << out[0].get_name()
           << ", count, \"" << n->description() << "\");\n";
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitAdd(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    writer << "{  // " << n->get_name() << "\n";
    writer.indent++;
    writer << "int count = " << out[0].get_size() << ";\n";
    writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
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
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitConcat(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitDot(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }

    const ngraph::op::Dot* dot = static_cast<const ngraph::op::Dot*>(n);
    const Shape& arg0_shape = args[0].get_shape();
    const Shape& arg1_shape = args[1].get_shape();
    if (arg0_shape.empty() || arg1_shape.empty())
    {
        auto& first = (arg0_shape.empty() ? args[0] : args[1]);
        auto& second = (arg0_shape.empty() ? args[1] : args[0]);
        writer << "{  // " << n->get_name() << "\n";
        writer.indent++;
        writer << "int count = " << second.get_size() << ";\n";
        writer << "cublasScopy("
               << "cublas_handle,"
               << "count ," << second.get_name() << ","
               << "1," << out[0].get_name() << ", 1);\n";
        writer << "cublasSscal("
               << "cublas_handle,"
               << "count ," << first.get_name() << "," << out[0].get_name() << ", 1);\n";
        writer.indent--;
        writer << "}\n";
        return;
    }

    //set output to 0 if input size is 0
    if (args[0].get_size() == 0 || args[1].get_size() == 0)
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << "runtime::gpu::cuda_memset(" << out[0].get_name() << ", 0, " << out[0].get_size()
               << " * sizeof(float));\n";
        writer.indent--;
        writer << "}\n";
        return;
    }

    if ((arg0_shape.size() == 1) && (arg1_shape.size() == 1))
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << "cublasSdot("
               << "cublas_handle," << arg0_shape[0] << "," << args[0].get_name() << ","
               << "1," << args[1].get_name() << ","
               << "1," << out[0].get_name() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 1))
    {
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
        writer << "const float alpha = 1.0;\n";
        writer << "const float beta  = 0;\n";
        writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);\n";
        writer << "cublasSgemv("
               << "cublas_handle,"
               << "CUBLAS_OP_T," << arg0_shape[0] << "," << arg0_shape[1] << ","
               << "&alpha," // Alpha
               << args[0].get_name() << "," << arg0_shape[1] << "," << args[1].get_name() << ","
               << "1,"
               << "&beta," // beta
               << out[0].get_name() << ","
               << "1);\n";
        writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);\n";
        writer.indent--;
        writer << "}\n";
    }
    else if ((arg0_shape.size() == 2) && (arg1_shape.size() == 2))
    {
        // GEMM Call
        if (arg0_shape[0] != out[0].get_shape()[0] || // m
            arg1_shape[1] != out[0].get_shape()[1] || // n
            arg0_shape[1] != arg1_shape[0])           // k
        {
            throw std::runtime_error("input and output shape is not correct for dot;");
        }
        writer << "{   // " << n->get_name() << "\n";
        writer.indent++;
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
        writer.indent--;
        writer << "}\n";
    }
    else
    {
        throw std::runtime_error(n->get_name() + " with more then 2D is not implemented.");
    }
}

void runtime::gpu::GPU_Emitter::EmitDivide(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitEqual(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitGreater(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitGreaterEq(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitLess(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitLessEq(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitMaximum(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    writer << "{  // " << n->get_name() << "\n";
    writer.indent++;
    writer << "int count = " << out[0].get_size() << ";\n";
    writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
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
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitMinimum(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    writer << "{  // " << n->get_name() << "\n";
    writer.indent++;
    writer << "int count = " << out[0].get_size() << ";\n";
    writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
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
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitNegative(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    writer << "{  // " << n->get_name() << "\n";
    writer.indent++;
    writer << "int count = " << out[0].get_size() << ";\n";
    writer += R"(
float alpha1 = -1.0, alpha2 = 0, beta = 0;
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
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
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitNotEqual(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}
void runtime::gpu::GPU_Emitter::EmitSelect(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitSubtract(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitBroadcast(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto broadcast = static_cast<const ngraph::op::Broadcast*>(n);
    auto arg_shape = args[0].get_shape();
    auto result_shape = out[0].get_shape();

    auto& axes = broadcast->get_broadcast_axes();
    //broadcast axes is empty, do a copy
    if (axes.empty())
    {
        writer << "{   // " << n->get_name() << " \n";
        writer.indent++;
        writer << "runtime::gpu::cuda_memcpyDtD(" << out[0].get_name() << ", " << args[0].get_name()
               << ", " << out[0].get_size() << " * " << out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
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

        writer << "{   // " << n->get_name() << " \n";
        writer.indent++;
        writer << "runtime::gpu::emit_broadcast(" << args[0].get_name() << ", " << out[0].get_name()
               << ", " << repeat_size << ", " << repeat_times << ", " << out[0].get_size()
               << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    else
    {
        throw std::runtime_error(n->get_name() + " is not implemented.");
    }
}

void runtime::gpu::GPU_Emitter::EmitConvert(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitConstant(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReshape(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    auto reshape = static_cast<const op::Reshape*>(n);
    writer << "{   // " << n->get_name() << "\n";
    writer.indent++;
    auto arg_shape = args[0].get_shape();
    auto arg_rank = arg_shape.size();

    auto result_shape = out[0].get_shape();
    auto& result_element_type = out[0].get_element_type();

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
        writer << "{   // " << n->get_name() << " 1\n";
        writer.indent++;
        writer << "runtime::gpu::cuda_memcpyDtD(" << out[0].get_name() << ", " << args[0].get_name()
               << ", " << out[0].get_size() << " * " << out[0].get_element_type().size() << ");\n";
        writer.indent--;
        writer << "}\n";
    }
    // If there *is* a layout change in the 2D case, we transpose the input.
    else if (arg_rank == 2)
    {
        // TODO Assert arg0_shape[0] == arg1_shape[0]?
        writer << "{   // " << n->get_name() << "\n";
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
               << args[0].get_name() << "," << arg_shape[1] << "," << out[0].get_name() << ","
               << result_shape[1] << ");\n";
        writer << "cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);\n";
        writer.indent--;
        writer << "}\n";
    }
    // Other cases (reordering of axes for tensors with rank>2) are not handled yet.
    else
    {
        throw runtime_error(
            "Axis permutation in reshape is not implemented yet for tensors with rank>2");
    }
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitFunctionCall(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
}

void runtime::gpu::GPU_Emitter::EmitReduce(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitSlice(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitSum(codegen::CodeWriter& writer,
                                        const ngraph::Node* n,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                        const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitMultiply(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    writer << "{  // " << n->get_name() << "\n";
    writer.indent++;
    writer << "int count = " << out[0].get_size() << ";\n";
    writer += R"(
float alpha1 = 1.0, alpha2 = 1.0, beta = 0;
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
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
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitPower(codegen::CodeWriter& writer,
                                          const ngraph::Node* n,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                          const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitReplaceSlice(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitOneHot(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitSqrt(codegen::CodeWriter& writer,
                                         const ngraph::Node* n,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                         const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    if (out[0].get_size() == 0)
    {
        return;
    }
    writer << "{  // " << n->get_name() << "\n";
    writer.indent++;
    writer << "int count = " << out[0].get_size() << ";\n";
    writer += R"(
float alpha1 = 1.0, alpha2 = 0, beta = 0;
cudnnTensorDescriptor_t descriptor;
cudnnCreateTensorDescriptor(&descriptor);
cudnnSetTensor4dDescriptor(descriptor,
                            /*format=*/CUDNN_TENSOR_NHWC,
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
    writer.indent--;
    writer << "}\n";
}

void runtime::gpu::GPU_Emitter::EmitConvolution(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitMaxPool(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitReverse(codegen::CodeWriter& writer,
                                            const ngraph::Node* n,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                            const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitReduceWindow(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitSelectAndScatter(
    codegen::CodeWriter& writer,
    const ngraph::Node* n,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
    const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    throw std::runtime_error(n->get_name() + " is not implemented.");
}

void runtime::gpu::GPU_Emitter::EmitResult(codegen::CodeWriter& writer,
                                           const ngraph::Node* n,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& args,
                                           const vector<runtime::gpu::GPU_TensorViewWrapper>& out)
{
    writer << "{   //" << n->get_name() << "\n";
    writer.indent++;
    writer << "runtime::gpu::cuda_memcpyDtD(" << out[0].get_name() << ", " << args[0].get_name()
           << ", " << out[0].get_size() << " * " << out[0].get_element_type().size() << ");\n";
    writer.indent--;
    writer << "}\n";
    return;
}
