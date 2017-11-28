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
#include "ngraph/runtime/interpreter/eigen/utils.hpp"
#include "ngraph/runtime/interpreter/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace eigen
            {
                template <typename T>
                class VectorSliceInstruction : public Instruction
                {
                public:
                    VectorSliceInstruction(const TensorViewInfo& arg,
                                           const TensorViewInfo& out,
                                           size_t lower,
                                           size_t upper)
                        : m_arg(arg)
                        , m_out(out)
                        , m_lower(lower)
                        , m_upper(upper)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        EigenVector<T>(out) =
                            EigenVector<T>(arg).segment(m_lower, m_upper - m_lower);
                    }

                protected:
                    TensorViewInfo m_arg;
                    TensorViewInfo m_out;
                    size_t m_lower;
                    size_t m_upper;
                };
            }
        }
    }
}
