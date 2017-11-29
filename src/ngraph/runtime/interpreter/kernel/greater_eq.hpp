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

#include "ngraph/runtime/interpreter/call_frame.hpp"
#include "ngraph/runtime/interpreter/kernel/utils.hpp"
#include "ngraph/runtime/interpreter/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace kernel
            {
                template <typename TI, typename TO>
                void greater_eq(TI arg0, TI arg1, TO out)
                {
                    auto result_as_float = get_map_array(&*arg0) <= get_map_array(&*arg1);
                    auto result_as_char = result_as_float.template cast<char>();
                    set_map_array(&*out, result_as_char);
                }

                template <typename T>
                class GreaterEqInstruction : public Instruction
                {
                public:
                    GreaterEqInstruction(TensorViewInfo arg0,
                                         TensorViewInfo arg1,
                                         TensorViewInfo out)
                        : m_arg0(arg0)
                        , m_arg1(arg1)
                        , m_out(out)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        EigenArray1d<element::Bool>(out) =
                            (EigenArray1d<T>(arg0) >=
                             EigenArray1d<T>(arg1))
                                .template cast<char>();
                    }

                protected:
                    TensorViewInfo m_arg0;
                    TensorViewInfo m_arg1;
                    TensorViewInfo m_out;
                };
            }
        }
    }
}
