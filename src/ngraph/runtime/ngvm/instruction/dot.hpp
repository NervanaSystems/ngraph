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

#include "ngraph/runtime/kernel/dot.hpp"
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
                class DotInstruction : public Instruction
                {
                public:
                    DotInstruction(const TensorViewInfo& arg0,
                                   const TensorViewInfo& arg1,
                                   const TensorViewInfo& out,
                                   const Shape& arg0_shape,
                                   const Shape& arg1_shape,
                                   const Shape& out_shape,
                                   size_t arg0_dot_axis,
                                   size_t arg1_dot_axis)
                        : m_arg0(arg0)
                        , m_arg1(arg1)
                        , m_out(out)
                        , m_arg0_shape(arg0_shape)
                        , m_arg1_shape(arg1_shape)
                        , m_out_shape(out_shape)
                        , m_arg0_dot_axis(arg0_dot_axis)
                        , m_arg1_dot_axis(arg1_dot_axis)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        typename ET::type* arg0 = get_tensor_data_ptr<ET>(call_frame, m_arg0);
                        typename ET::type* arg1 = get_tensor_data_ptr<ET>(call_frame, m_arg1);
                        typename ET::type* out = get_tensor_data_ptr<ET>(call_frame, m_out);

                        kernel::dot<typename ET::type>(arg0,
                                                       arg1,
                                                       out,
                                                       m_arg0_shape,
                                                       m_arg1_shape,
                                                       m_out_shape,
                                                       m_arg0_dot_axis,
                                                       m_arg1_dot_axis);
                    }

                protected:
                    TensorViewInfo m_arg0;
                    TensorViewInfo m_arg1;
                    TensorViewInfo m_out;
                    Shape m_arg0_shape;
                    Shape m_arg1_shape;
                    Shape m_out_shape;
                    size_t m_arg0_dot_axis;
                    size_t m_arg1_dot_axis;
                };
            }
        }
    }
}
