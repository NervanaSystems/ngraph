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
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"

#define EMITTER_DECL(op_name)                                                                      \
    emit<op_name>(CPU_ExternalFunction * external_function,                                        \
                  codegen::CodeWriter & writer,                                                    \
                  const ngraph::Node* node,                                                        \
                  const std::vector<TensorViewWrapper>& args,                                      \
                  const std::vector<TensorViewWrapper>& out)

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_Emitter
            {
            public:
                template <typename OP>
                static void emit(CPU_ExternalFunction* external_function,
                                 codegen::CodeWriter& writer,
                                 const ngraph::Node* node,
                                 const std::vector<TensorViewWrapper>& args,
                                 const std::vector<TensorViewWrapper>& out)
                {
                    throw std::runtime_error("Unimplemented op in CPU emitter");
                }

                static void nop(CPU_ExternalFunction* external_function,
                                codegen::CodeWriter& writer,
                                const ngraph::Node* node,
                                const std::vector<TensorViewWrapper>& args,
                                const std::vector<TensorViewWrapper>& out)
                {
                }

                static void emitBatchNorm(CPU_ExternalFunction* external_function,
                                          codegen::CodeWriter& writer,
                                          const ngraph::Node* node,
                                          const std::vector<TensorViewWrapper>& args,
                                          const std::vector<TensorViewWrapper>& out,
                                          bool append_relu = false);

            private:
                static std::string emit_vector(const TensorViewWrapper&,
                                               const std::string& name = "");
                static std::string emit_array1d(const TensorViewWrapper&,
                                                const std::string& name = "");
                static std::string emit_matrix(const TensorViewWrapper&,
                                               const std::string& name = "");

                static std::string emit_for_lt(const std::string& prefix, size_t index, size_t to);
                static std::string emit_indices(const std::vector<std::string> indices);
            };
        }
    }
}
