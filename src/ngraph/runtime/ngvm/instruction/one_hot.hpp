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

#include "ngraph/runtime/kernel/one_hot.hpp"
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
                class OneHotInstruction : public Instruction
                {
                public:
                    OneHotInstruction(const TensorViewInfo& arg,
                                      const TensorViewInfo& out,
                                      const Shape& arg_shape,
                                      const Shape& out_shape,
                                      size_t one_hot_axis)
                        : m_arg(arg)
                        , m_out(out)
                        , m_arg_shape(arg_shape)
                        , m_out_shape(out_shape)
                        , m_one_hot_axis(one_hot_axis)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        typename ET::type* arg = get_tensor_data_ptr<ET>(call_frame, m_arg);
                        typename ET::type* out = get_tensor_data_ptr<ET>(call_frame, m_out);

                        kernel::one_hot<typename ET::type>(
                            arg, out, m_arg_shape, m_out_shape, m_one_hot_axis);
                    }

                protected:
                    TensorViewInfo m_arg;
                    TensorViewInfo m_out;
                    Shape m_arg_shape;
                    Shape m_out_shape;
                    size_t m_one_hot_axis;
                };
            }
        }
    }
}
