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

#pragma once

#include <string>
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp"

#define EMITTER_DECL(op_name)                                                                      \
    emit<op_name>(GPU_ExternalFunction * external_function,                                        \
                  codegen::CodeWriter & writer,                                                    \
                  const ngraph::Node* node,                                                        \
                  const std::vector<GPU_TensorViewWrapper>& args,                                  \
                  const std::vector<GPU_TensorViewWrapper>& out)
namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter
            {
            public:
                template <typename OP>
                static void emit(GPU_ExternalFunction* external_function,
                                 codegen::CodeWriter& writer,
                                 const ngraph::Node* node,
                                 const std::vector<GPU_TensorViewWrapper>& args,
                                 const std::vector<GPU_TensorViewWrapper>& out)
                {
                    throw std::runtime_error("Unimplemented op in GPU emitter for " +
                                             node->get_name());
                }

                static void nop(GPU_ExternalFunction* external_function,
                                codegen::CodeWriter& writer,
                                const ngraph::Node* node,
                                const std::vector<GPU_TensorViewWrapper>& args,
                                const std::vector<GPU_TensorViewWrapper>& out)
                {
                }

                static void emit_elementwise(GPU_ExternalFunction* external_function,
                                             codegen::CodeWriter& writer,
                                             const ngraph::Node* node,
                                             const std::vector<GPU_TensorViewWrapper>& args,
                                             const std::vector<GPU_TensorViewWrapper>& out);
            };
            Shape get_padded_shape(const Shape& input_shape,
                                   const Shape& padding_below,
                                   const Shape& padding_above,
                                   const Shape& padding_interior);
        }
    }
}
