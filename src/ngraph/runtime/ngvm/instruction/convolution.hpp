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

#include "ngraph/runtime/kernel/convolution.hpp"
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
                class ConvolutionInstruction : public Instruction
                {
                public:
                    ConvolutionInstruction(const TensorViewInfo& arg0,
                                           const TensorViewInfo& arg1,
                                           const TensorViewInfo& out,
                                           const Shape& arg0_shape,
                                           const Shape& arg1_shape,
                                           const Shape& out_shape,
                                           const Strides& window_movement_strides,
                                           const Strides& window_dilation_strides)
                        : m_arg0(arg0)
                        , m_arg1(arg1)
                        , m_out(out)
                        , m_arg0_shape(arg0_shape)
                        , m_arg1_shape(arg1_shape)
                        , m_out_shape(out_shape)
                        , m_window_movement_strides(window_movement_strides)
                        , m_window_dilation_strides(window_dilation_strides)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        typename ET::type* arg0 = get_tensor_data_ptr<ET>(call_frame, m_arg0);
                        typename ET::type* arg1 = get_tensor_data_ptr<ET>(call_frame, m_arg1);
                        typename ET::type* out = get_tensor_data_ptr<ET>(call_frame, m_out);

                        kernel::convolution<typename ET::type>(arg0,
                                                               arg1,
                                                               out,
                                                               m_arg0_shape,
                                                               m_arg1_shape,
                                                               m_out_shape,
                                                               m_window_movement_strides,
                                                               m_window_dilation_strides);
                    }

                protected:
                    TensorViewInfo m_arg0;
                    TensorViewInfo m_arg1;
                    TensorViewInfo m_out;
                    Shape m_arg0_shape;
                    Shape m_arg1_shape;
                    Shape m_out_shape;
                    Strides m_window_movement_strides;
                    Strides m_window_dilation_strides;
                };
            }
        }
    }
}
