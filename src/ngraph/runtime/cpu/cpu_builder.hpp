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

#include "ngraph/node.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"

#define BUILDER_DECL(op_name)                                                                      \
    build<op_name>(CPU_ExternalFunction * external_function,                                       \
                   const ngraph::Node* node,                                                       \
                   const std::vector<TensorViewWrapper>& args,                                     \
                   const std::vector<TensorViewWrapper>& out)

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            using BuildOpFunction =
                std::function<void(CPU_ExternalFunction* external_function,
                                   const ngraph::Node*,
                                   const std::vector<TensorViewWrapper>& inputs,
                                   const std::vector<TensorViewWrapper>& outputs)>;

            using BuildOpMap = std::unordered_map<std::type_index, BuildOpFunction>;

            extern const BuildOpMap build_dispatcher;

            class Builder
            {
            public:
                template <typename OP>
                static void build(CPU_ExternalFunction* external_function,
                                  const ngraph::Node* node,
                                  const std::vector<TensorViewWrapper>& args,
                                  const std::vector<TensorViewWrapper>& out)
                {
                    throw std::runtime_error("Unimplemented op in CPU builder");
                }

                static void nop(CPU_ExternalFunction* external_function,
                                const ngraph::Node* node,
                                const std::vector<TensorViewWrapper>& args,
                                const std::vector<TensorViewWrapper>& out)
                {
                }

                static void buildBatchNorm(CPU_ExternalFunction* external_function,
                                           const ngraph::Node* node,
                                           const std::vector<TensorViewWrapper>& args,
                                           const std::vector<TensorViewWrapper>& out,
                                           bool append_relu = false);
            };
        }
    }
}
