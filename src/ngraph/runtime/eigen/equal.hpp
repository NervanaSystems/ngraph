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

#include "ngraph/runtime/call_frame.hpp"
#include "ngraph/runtime/eigen/utils.hpp"
#include "ngraph/runtime/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
            template <typename TI,typename TO>
            void equal(TI arg0, TI arg1, TO out)
            {
                auto result_as_float = get_map(&*arg0) == get_map(&*arg1);
                auto result_as_char  = result_as_float.template cast<char>();
                set_map(&*out, result_as_char);
            }

            template <typename ET>
            class EqualInstruction : public Instruction
            {
            public:
                EqualInstruction(size_t arg0, size_t arg1, size_t out)
                    : m_arg0(arg0)
                    , m_arg1(arg1)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    runtime::eigen::equal(
                        call_frame.get_parameterized_tensor<ET>(m_arg0),
                        call_frame.get_parameterized_tensor<ET>(m_arg1),
                        call_frame.get_parameterized_tensor<element::Bool>(m_out));
                }

            protected:
                size_t m_arg0;
                size_t m_arg1;
                size_t m_out;
            };
        }
    }
}
