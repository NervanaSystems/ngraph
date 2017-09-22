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
            template <typename T>
            void negate(T* arg, T* out)
            {
                set_map(out, -(get_map(arg)));
            }

            template <typename T>
            void negate(std::shared_ptr<T>& arg, std::shared_ptr<T>& out)
            {
                negate(&*arg, &*out);
            }

            template <typename ET>
            class NegateInstruction : public Instruction
            {
            public:
                NegateInstruction(size_t arg, size_t out)
                    : m_arg(arg)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    negate(call_frame.get_parameterized_tensor<ET>(m_arg),
                           call_frame.get_parameterized_tensor<ET>(m_out));
                }

            protected:
                size_t m_arg;
                size_t m_out;
            };
        }
    }
}
