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
            template <typename TA,typename TB>
            void select(TA arg0, TB arg1, TB arg2, TB out)
            {
                set_map(&*out, get_map(&*arg0).select(get_map(&*arg1),get_map(&*arg2)));
            }

            template <typename ET>
            class SelectInstruction : public Instruction
            {
            public:
                SelectInstruction(size_t arg0, size_t arg1, size_t arg2, size_t out)
                    : m_arg0(arg0)
                    , m_arg1(arg1)
                    , m_arg2(arg2)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    runtime::eigen::select(
                        call_frame.get_parameterized_tensor<element::Bool>(m_arg0),
                        call_frame.get_parameterized_tensor<ET>(m_arg1),
                        call_frame.get_parameterized_tensor<ET>(m_arg2),
                        call_frame.get_parameterized_tensor<ET>(m_out));
                }

            protected:
                size_t m_arg0;
                size_t m_arg1;
                size_t m_arg2;
                size_t m_out;
            };
        }
    }
}
