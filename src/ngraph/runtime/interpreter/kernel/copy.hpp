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

#include <cassert>

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
                /// @brief Copies a tensor from in to out.
                template <typename T>
                class CopyInstruction : public Instruction
                {
                public:
                    /// @param in Index of input tensor in call frame.
                    /// @param out Index of output tensor in call frame.
                    CopyInstruction(size_t in, size_t out)
                        : m_in(in)
                        , m_out(out)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        call_frame.get_parameterized_tensor_view<T>(m_out)->write(
                            call_frame.get_parameterized_tensor_view<T>(m_in)->get_vector());
                    }

                protected:
                    size_t m_in;
                    size_t m_out;
                };
            }
        }
    }
}
