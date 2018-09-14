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

#pragma once

#include <string>
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter;

            Shape get_padded_shape(const Shape& input_shape,
                                   const Shape& padding_below,
                                   const Shape& padding_above,
                                   const Shape& padding_interior);
        }
    }
}

class ngraph::runtime::gpu::GPU_Emitter
{
public:
    static void emit_Abs(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Acos(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Add(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_AllReduce(GPU_ExternalFunction* external_function,
                               codegen::CodeWriter& writer,
                               const ngraph::Node* node,
                               const std::vector<GPU_TensorViewWrapper>& args,
                               const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_And(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ArgMax(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ArgMin(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Asin(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Atan(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_AvgPool(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_AvgPoolBackprop(GPU_ExternalFunction* external_function,
                                     codegen::CodeWriter& writer,
                                     const ngraph::Node* node,
                                     const std::vector<GPU_TensorViewWrapper>& args,
                                     const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_BatchNorm(GPU_ExternalFunction* external_function,
                               codegen::CodeWriter& writer,
                               const ngraph::Node* node,
                               const std::vector<GPU_TensorViewWrapper>& args,
                               const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_BatchNormBackprop(GPU_ExternalFunction* external_function,
                                       codegen::CodeWriter& writer,
                                       const ngraph::Node* node,
                                       const std::vector<GPU_TensorViewWrapper>& args,
                                       const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Broadcast(GPU_ExternalFunction* external_function,
                               codegen::CodeWriter& writer,
                               const ngraph::Node* node,
                               const std::vector<GPU_TensorViewWrapper>& args,
                               const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Ceiling(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Concat(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Constant(GPU_ExternalFunction* external_function,
                              codegen::CodeWriter& writer,
                              const ngraph::Node* node,
                              const std::vector<GPU_TensorViewWrapper>& args,
                              const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Convert(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Convolution(GPU_ExternalFunction* external_function,
                                 codegen::CodeWriter& writer,
                                 const ngraph::Node* node,
                                 const std::vector<GPU_TensorViewWrapper>& args,
                                 const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ConvolutionBackpropData(GPU_ExternalFunction* external_function,
                                             codegen::CodeWriter& writer,
                                             const ngraph::Node* node,
                                             const std::vector<GPU_TensorViewWrapper>& args,
                                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ConvolutionBackpropFilters(GPU_ExternalFunction* external_function,
                                                codegen::CodeWriter& writer,
                                                const ngraph::Node* node,
                                                const std::vector<GPU_TensorViewWrapper>& args,
                                                const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Cos(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Cosh(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Divide(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Dot(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Equal(GPU_ExternalFunction* external_function,
                           codegen::CodeWriter& writer,
                           const ngraph::Node* node,
                           const std::vector<GPU_TensorViewWrapper>& args,
                           const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Exp(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Floor(GPU_ExternalFunction* external_function,
                           codegen::CodeWriter& writer,
                           const ngraph::Node* node,
                           const std::vector<GPU_TensorViewWrapper>& args,
                           const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_FunctionCall(GPU_ExternalFunction* external_function,
                                  codegen::CodeWriter& writer,
                                  const ngraph::Node* node,
                                  const std::vector<GPU_TensorViewWrapper>& args,
                                  const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_GetOutputElement(GPU_ExternalFunction* external_function,
                                      codegen::CodeWriter& writer,
                                      const ngraph::Node* node,
                                      const std::vector<GPU_TensorViewWrapper>& args,
                                      const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Greater(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_GreaterEq(GPU_ExternalFunction* external_function,
                               codegen::CodeWriter& writer,
                               const ngraph::Node* node,
                               const std::vector<GPU_TensorViewWrapper>& args,
                               const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Less(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_LessEq(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Log(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_LRN(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Max(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Maximum(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_MaxPool(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_MaxPoolBackprop(GPU_ExternalFunction* external_function,
                                     codegen::CodeWriter& writer,
                                     const ngraph::Node* node,
                                     const std::vector<GPU_TensorViewWrapper>& args,
                                     const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Min(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Minimum(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Multiply(GPU_ExternalFunction* external_function,
                              codegen::CodeWriter& writer,
                              const ngraph::Node* node,
                              const std::vector<GPU_TensorViewWrapper>& args,
                              const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Negative(GPU_ExternalFunction* external_function,
                              codegen::CodeWriter& writer,
                              const ngraph::Node* node,
                              const std::vector<GPU_TensorViewWrapper>& args,
                              const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Not(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_NotEqual(GPU_ExternalFunction* external_function,
                              codegen::CodeWriter& writer,
                              const ngraph::Node* node,
                              const std::vector<GPU_TensorViewWrapper>& args,
                              const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_OneHot(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Or(GPU_ExternalFunction* external_function,
                        codegen::CodeWriter& writer,
                        const ngraph::Node* node,
                        const std::vector<GPU_TensorViewWrapper>& args,
                        const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Pad(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Parameter(GPU_ExternalFunction* external_function,
                               codegen::CodeWriter& writer,
                               const ngraph::Node* node,
                               const std::vector<GPU_TensorViewWrapper>& args,
                               const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Power(GPU_ExternalFunction* external_function,
                           codegen::CodeWriter& writer,
                           const ngraph::Node* node,
                           const std::vector<GPU_TensorViewWrapper>& args,
                           const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Product(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Reduce(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ReduceWindow(GPU_ExternalFunction* external_function,
                                  codegen::CodeWriter& writer,
                                  const ngraph::Node* node,
                                  const std::vector<GPU_TensorViewWrapper>& args,
                                  const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Relu(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ReluBackprop(GPU_ExternalFunction* external_function,
                                  codegen::CodeWriter& writer,
                                  const ngraph::Node* node,
                                  const std::vector<GPU_TensorViewWrapper>& args,
                                  const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ReplaceSlice(GPU_ExternalFunction* external_function,
                                  codegen::CodeWriter& writer,
                                  const ngraph::Node* node,
                                  const std::vector<GPU_TensorViewWrapper>& args,
                                  const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Reshape(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Result(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Reverse(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_ReverseSequence(GPU_ExternalFunction* external_function,
                                     codegen::CodeWriter& writer,
                                     const ngraph::Node* node,
                                     const std::vector<GPU_TensorViewWrapper>& args,
                                     const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Select(GPU_ExternalFunction* external_function,
                            codegen::CodeWriter& writer,
                            const ngraph::Node* node,
                            const std::vector<GPU_TensorViewWrapper>& args,
                            const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_SelectAndScatter(GPU_ExternalFunction* external_function,
                                      codegen::CodeWriter& writer,
                                      const ngraph::Node* node,
                                      const std::vector<GPU_TensorViewWrapper>& args,
                                      const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Sigmoid(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_SigmoidBackprop(GPU_ExternalFunction* external_function,
                                     codegen::CodeWriter& writer,
                                     const ngraph::Node* node,
                                     const std::vector<GPU_TensorViewWrapper>& args,
                                     const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Sign(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Sin(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Sinh(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Slice(GPU_ExternalFunction* external_function,
                           codegen::CodeWriter& writer,
                           const ngraph::Node* node,
                           const std::vector<GPU_TensorViewWrapper>& args,
                           const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Softmax(GPU_ExternalFunction* external_function,
                             codegen::CodeWriter& writer,
                             const ngraph::Node* node,
                             const std::vector<GPU_TensorViewWrapper>& args,
                             const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Sqrt(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_StopGradient(GPU_ExternalFunction* external_function,
                                  codegen::CodeWriter& writer,
                                  const ngraph::Node* node,
                                  const std::vector<GPU_TensorViewWrapper>& args,
                                  const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Subtract(GPU_ExternalFunction* external_function,
                              codegen::CodeWriter& writer,
                              const ngraph::Node* node,
                              const std::vector<GPU_TensorViewWrapper>& args,
                              const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Sum(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Tan(GPU_ExternalFunction* external_function,
                         codegen::CodeWriter& writer,
                         const ngraph::Node* node,
                         const std::vector<GPU_TensorViewWrapper>& args,
                         const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_Tanh(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);
    static void emit_TopK(GPU_ExternalFunction* external_function,
                          codegen::CodeWriter& writer,
                          const ngraph::Node* node,
                          const std::vector<GPU_TensorViewWrapper>& args,
                          const std::vector<GPU_TensorViewWrapper>& out);

    template <typename T>
    static void emit_elementwise(GPU_ExternalFunction* external_function,
                                 codegen::CodeWriter& writer,
                                 const ngraph::Node* node,
                                 const std::vector<GPU_TensorViewWrapper>& args,
                                 const std::vector<GPU_TensorViewWrapper>& out)
    {
        if (out[0].get_size() == 0)
        {
            return;
        }
        else if (out.size() > 1)
        {
            throw std::runtime_error("Multi-output elementwise ops are not currently supported.");
        }
        auto& cuda_emitter = external_function->get_primitive_emitter()->get_cuda_emitter();

        writer.block_begin();
        {
            std::vector<std::string> dtypes;
            for (auto& arg : args)
            {
                dtypes.push_back(arg.get_type());
            }
            dtypes.push_back(out[0].get_type());
            auto ew_index = cuda_emitter->build_elementwise<T>(dtypes, out[0].get_shape());
            writer << "gpu::invoke_primitive(ctx, " << ew_index << ", ";
            writer << "std::vector<void*>{" << args.front().get_name();
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
};
