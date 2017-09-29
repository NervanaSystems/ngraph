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
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
            template <typename T>
            void abs(T* arg, const TH2& arg_th, T* out, const TH2& out_th)
            {
                set_map_array(out, out_th, Eigen::abs(get_map_array(arg, arg_th)));
            }

            template <typename ET>
            class AbsInstruction : public Instruction
            {
            public:
                AbsInstruction(const TensorViewInfo& arg, const TensorViewInfo& out)
                    : m_arg(get_tensor_header(arg, true))
                    , m_out(get_tensor_header(out, true))
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    runtime::eigen::abs(
                        call_frame.get_tensor_view_data<ET>(m_arg.index),
                        m_arg,
                        call_frame.get_tensor_view_data<ET>(m_out.index),
                        m_out);
                }

            protected:
                TH2 m_arg;
                TH2 m_out;
            };
        }
    }
}
