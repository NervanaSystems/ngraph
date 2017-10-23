// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <vector>
#include <string>

#include "ngraph/node.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"
#include "ngraph/runtime/cpu/external_function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class Emitter
            {
            protected:
                std::string TU;

            public:
                Emitter() : TU("") { }
                std::string& GetTU() { return TU; }
                
                void EmitNop(const ngraph::Node*,
                             ExternalFunction*,
                             FunctionMap&,
                             const std::vector<TensorViewInfo>& inputs,
                             const std::vector<TensorViewInfo>& outputs);

                void EmitAdd(const ngraph::Node*,
                             ExternalFunction*,
                             FunctionMap&,
                             const std::vector<TensorViewInfo>& inputs,
                             const std::vector<TensorViewInfo>& outputs);

                void EmitDot(const ngraph::Node*,
                             ExternalFunction*,
                             FunctionMap&,
                             const std::vector<TensorViewInfo>& inputs,
                             const std::vector<TensorViewInfo>& outputs);

                void EmitMultiply(const ngraph::Node*,
                                  ExternalFunction*,
                                  FunctionMap&,
                                  const std::vector<TensorViewInfo>& inputs,
                                  const std::vector<TensorViewInfo>& outputs);

                void EmitGetTupleElement(const ngraph::Node*,
                                         ExternalFunction*,
                                         FunctionMap&,
                                         const std::vector<TensorViewInfo>& inputs,
                                         const std::vector<TensorViewInfo>& outputs);

                void EmitTuple(const ngraph::Node*,
                               ExternalFunction*,
                               FunctionMap&,
                               const std::vector<TensorViewInfo>& inputs,
                               const std::vector<TensorViewInfo>& outputs);

                void EmitAbs(const ngraph::Node*,
                             ExternalFunction*,
                             FunctionMap&,
                             const std::vector<TensorViewInfo>& inputs,
                             const std::vector<TensorViewInfo>& outputs);

            };
        }
    }
}
