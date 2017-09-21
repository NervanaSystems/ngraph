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

#include "runtime/call_frame.hpp"
#include "runtime/eigen/utils.hpp"
#include "runtime/instruction.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
            template <typename T>
            void add(T* arg0, T* arg1, T* out)
            {
                set_map(out, get_map(arg0) + get_map(arg1));
            }

            template <typename T>
            void add(std::shared_ptr<T>& arg0, std::shared_ptr<T>& arg1, std::shared_ptr<T>& out)
            {
                add(&*arg0, &*arg1, &*out);
            }

            template <typename ET>
            class AddInstruction : public Instruction
            {
            public:
                AddInstruction(size_t arg0, size_t arg1, size_t out)
                    : m_arg0(arg0)
                    , m_arg1(arg1)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    add(call_frame.get_parameterized_tensor<ET>(m_arg0),
                        call_frame.get_parameterized_tensor<ET>(m_arg1),
                        call_frame.get_parameterized_tensor<ET>(m_out));
                }

            protected:
                size_t m_arg0;
                size_t m_arg1;
                size_t m_out;
            };
        }
    }
}
