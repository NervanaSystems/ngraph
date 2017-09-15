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
                    dynamic_cast<PrimaryTensorView<ET>*>(&*call_frame.get_tensor(m_out))
                        ->get_map() =
                        dynamic_cast<PrimaryTensorView<ET>*>(&*call_frame.get_tensor(m_arg0))
                            ->get_map() +
                        dynamic_cast<PrimaryTensorView<ET>*>(&*call_frame.get_tensor(m_arg1))
                            ->get_map();
                }

            protected:
                size_t m_arg0;
                size_t m_arg1;
                size_t m_out;
            };
        }
    }
}
