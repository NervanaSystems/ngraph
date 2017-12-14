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

#include "ngraph/runtime/ngvm/call_frame.hpp"
#include "ngraph/runtime/ngvm/instruction.hpp"
#include "ngraph/runtime/ngvm/tensor_view_info.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ngvm
        {
            namespace instruction
            {
                template <typename ET>
                class ConstantInstruction : public Instruction
                {
                public:
                    ConstantInstruction(const std::vector<typename ET::type> value,
                                        const TensorViewInfo& out)
                        : m_value(value)
                        , m_out(out)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        call_frame.get_tensor_view(m_out.get_index())->write(m_value);
                    }

                protected:
                    const std::vector<typename ET::type> m_value;
                    TensorViewInfo m_out;
                };
            }
        }
    }
}
