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

#include "ngraph/runtime/kernel/copy.hpp"
#include "ngraph/runtime/ngvm/call_frame.hpp"
#include "ngraph/runtime/ngvm/instruction.hpp"
#include "ngraph/runtime/ngvm/utils.hpp"
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
                class CopyInstruction : public Instruction
                {
                public:
                    CopyInstruction(const TensorViewInfo& arg, const TensorViewInfo& out)
                        : m_arg(arg)
                        , m_out(out)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        typename ET::type* arg = get_tensor_data_ptr<ET>(call_frame, m_arg);
                        typename ET::type* out = get_tensor_data_ptr<ET>(call_frame, m_out);

                        size_t count = get_tensor_element_count(call_frame, m_arg);

                        kernel::copy<typename ET::type>(arg, out, count);
                    }

                protected:
                    TensorViewInfo m_arg;
                    TensorViewInfo m_out;
                };
            }
        }
    }
}
