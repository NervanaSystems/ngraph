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
#include "ngraph/runtime/eigen/tensor_view.hpp"
#include "ngraph/runtime/instruction.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
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
                    dynamic_cast<PrimaryTensorView<ET>*>(&*call_frame.get_tensor(m_out))
                        ->get_map() =
                          -(dynamic_cast<PrimaryTensorView<ET>*>(&*call_frame.get_tensor(m_arg))
                              ->get_map());
                }

            protected:
                size_t m_arg;
                size_t m_out;
            };
        }
    }
}
