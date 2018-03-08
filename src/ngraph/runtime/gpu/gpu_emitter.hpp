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

#define EMITTER_DECL(E)                                                                            \
    E(codegen::CodeWriter& writer,                                                                 \
      const ngraph::Node* n,                                                                       \
      const std::vector<ngraph::runtime::gpu::GPU_TensorViewWrapper>& args,                        \
      const std::vector<ngraph::runtime::gpu::GPU_TensorViewWrapper>& out)

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter
            {
            public:
                static void EMITTER_DECL(EmitNop);
                static void EMITTER_DECL(EmitAdd);
                static void EMITTER_DECL(EmitDot);
                static void EMITTER_DECL(EmitMultiply);
                static void EMITTER_DECL(EmitGetOutputElement);
                static void EMITTER_DECL(EmitXLAGetTupleElement);
                static void EMITTER_DECL(EmitUnaryElementwise);
                static void EMITTER_DECL(EmitTuple);
                static void EMITTER_DECL(EmitConcat);
                static void EMITTER_DECL(EmitDivide);
                static void EMITTER_DECL(EmitEqual);
                static void EMITTER_DECL(EmitGreater);
                static void EMITTER_DECL(EmitGreaterEq);
                static void EMITTER_DECL(EmitLess);
                static void EMITTER_DECL(EmitLessEq);
                static void EMITTER_DECL(EmitMaximum);
                static void EMITTER_DECL(EmitMinimum);
                static void EMITTER_DECL(EmitNegative);
                static void EMITTER_DECL(EmitNotEqual);
                static void EMITTER_DECL(EmitSelect);
                static void EMITTER_DECL(EmitSubtract);
                static void EMITTER_DECL(EmitBroadcast);
                static void EMITTER_DECL(EmitConvert);
                static void EMITTER_DECL(EmitConstant);
                static void EMITTER_DECL(EmitReshape);
                static void EMITTER_DECL(EmitFunctionCall);
                static void EMITTER_DECL(EmitReduce);
                static void EMITTER_DECL(EmitSlice);
                static void EMITTER_DECL(EmitSum);
                static void EMITTER_DECL(EmitPower);
                static void EMITTER_DECL(EmitReplaceSlice);
                static void EMITTER_DECL(EmitOneHot);
                static void EMITTER_DECL(EmitSqrt);
                static void EMITTER_DECL(EmitConvolution);
                static void EMITTER_DECL(EmitMaxPool);
                static void EMITTER_DECL(EmitReverse);
                static void EMITTER_DECL(EmitReduceWindow);
                static void EMITTER_DECL(EmitSelectAndScatter);
                static void EMITTER_DECL(EmitResult);
            };
        }
    }
}
