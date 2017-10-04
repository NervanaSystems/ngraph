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
            template <typename ET>
            class BroadcastScalarInstruction : public Instruction
            {
            public:
                BroadcastScalarInstruction(const TensorViewInfo& arg, const TensorViewInfo& out)
                    : m_arg(arg)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    // This is a bit hacky: regardless of the tensor rank we
                    // pull it out as a vector. This works because of the way
                    // fmt::V computes sizes---it lumps together any higher
                    // dimensions---while fmt::M ignores them.
                    EigenArray1d<ET>(call_frame, m_out) = EigenArray1d<ET>(call_frame, m_arg)(0, 0);
                }

            protected:
                TensorViewInfo m_arg;
                TensorViewInfo m_out;
            };
        }
    }
}
