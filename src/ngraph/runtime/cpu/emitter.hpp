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

#define EMITTER_DECL(E) E(const ngraph::Node* n,                     \
                          ExternalFunction* ef,                      \
                          FunctionMap& function_map,                 \
                          const std::vector<TensorViewInfo>& inputs, \
                          const std::vector<TensorViewInfo>& outputs)


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

                void EMITTER_DECL(EmitNop);
                void EMITTER_DECL(EmitAdd);
                void EMITTER_DECL(EmitDot);
                void EMITTER_DECL(EmitMultiply);
                void EMITTER_DECL(EmitGetTupleElement);
                void EMITTER_DECL(EmitTuple);
                void EMITTER_DECL(EmitAbs);
                void EMITTER_DECL(EmitConcat);
                void EMITTER_DECL(EmitDivide);
                void EMITTER_DECL(EmitEqual);
                void EMITTER_DECL(EmitGreater);
                void EMITTER_DECL(EmitGreaterEq);
                void EMITTER_DECL(EmitLess);
                void EMITTER_DECL(EmitLessEq);

            };
        }
    }
}
