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

#include <string>
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view_wrapper.hpp"

#define EMITTER_DECL(E)                                                                            \
    E(const ngraph::Node* n,                                                                       \
      const std::vector<ngraph::runtime::cpu::TensorViewWrapper>& args,                            \
      const std::vector<ngraph::runtime::cpu::TensorViewWrapper>& out)

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            class CPU_Emitter
            {
            protected:
                codegen::CodeWriter m_out;
                bool m_use_ref_kernels;

            public:
                CPU_Emitter()
                    : m_out()
                    , m_use_ref_kernels(std::getenv("NGRAPH_CPU_USE_REF_KERNELS") != nullptr)
                {
                }
                std::string get_code() { return m_out.get_code(); }
                codegen::CodeWriter& get_code_writer() { return m_out; }
                void EMITTER_DECL(EmitNop);
                void EMITTER_DECL(EmitAdd);
                void EMITTER_DECL(EmitDot);
                void EMITTER_DECL(EmitMultiply);
                void EMITTER_DECL(EmitGetOutputElement);
                void EMITTER_DECL(EmitXLAGetTupleElement);
                void EMITTER_DECL(EmitTuple);
                void EMITTER_DECL(EmitAbs);
                void EMITTER_DECL(EmitConcat);
                void EMITTER_DECL(EmitDivide);
                void EMITTER_DECL(EmitEqual);
                void EMITTER_DECL(EmitGreater);
                void EMITTER_DECL(EmitGreaterEq);
                void EMITTER_DECL(EmitLess);
                void EMITTER_DECL(EmitLessEq);
                void EMITTER_DECL(EmitLog);
                void EMITTER_DECL(EmitMaximum);
                void EMITTER_DECL(EmitMinimum);
                void EMITTER_DECL(EmitNegative);
                void EMITTER_DECL(EmitNotEqual);
                void EMITTER_DECL(EmitSelect);
                void EMITTER_DECL(EmitSubtract);
                void EMITTER_DECL(EmitParameterizedConstantBool);
                void EMITTER_DECL(EmitParameterizedConstantFloat32);
                void EMITTER_DECL(EmitParameterizedConstantInt8);
                void EMITTER_DECL(EmitParameterizedConstantInt32);
                void EMITTER_DECL(EmitParameterizedConstantInt64);
                void EMITTER_DECL(EmitParameterizedConstantUInt8);
                void EMITTER_DECL(EmitParameterizedConstantUInt32);
                void EMITTER_DECL(EmitParameterizedConstantUInt64);
                void EMITTER_DECL(EmitBroadcast);
                void EMITTER_DECL(EmitConvert);
                void EMITTER_DECL(EmitConstant);
                void EMITTER_DECL(EmitReshape);
                void EMITTER_DECL(EmitFunctionCall);
                void EMITTER_DECL(EmitReduce);
                void EMITTER_DECL(EmitSign);
                void EMITTER_DECL(EmitSlice);
                void EMITTER_DECL(EmitSum);
                void EMITTER_DECL(EmitExp);
                void EMITTER_DECL(EmitSin);
                void EMITTER_DECL(EmitSinh);
                void EMITTER_DECL(EmitCos);
                void EMITTER_DECL(EmitCosh);
                void EMITTER_DECL(EmitTan);
                void EMITTER_DECL(EmitTanh);
                void EMITTER_DECL(EmitAsin);
                void EMITTER_DECL(EmitAcos);
                void EMITTER_DECL(EmitAtan);
                void EMITTER_DECL(EmitPower);
                void EMITTER_DECL(EmitReplaceSlice);
                void EMITTER_DECL(EmitOneHot);
                void EMITTER_DECL(EmitFloor);
                void EMITTER_DECL(EmitCeiling);
                void EMITTER_DECL(EmitSqrt);
                void EMITTER_DECL(EmitConvolution);

            private:
                void generate_call(const std::vector<TensorViewWrapper>& args,
                                   const std::vector<TensorViewWrapper>& out,
                                   std::shared_ptr<Function> function);

                std::string emit_vector(const TensorViewWrapper&, const std::string& name = "");
                std::string emit_array1d(const TensorViewWrapper&, const std::string& name = "");
                std::string emit_matrix(const TensorViewWrapper&, const std::string& name = "");
            };
        }
    }
}
