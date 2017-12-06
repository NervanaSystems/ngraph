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

#include <vector>

#include "ngraph/runtime/kernel/concat.hpp"
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
                class ConcatInstruction : public Instruction
                {
                public:
                    ConcatInstruction(const std::vector<TensorViewInfo>& args,
                                      const TensorViewInfo& out,
                                      const std::vector<Shape>& arg_shapes,
                                      const Shape& out_shape,
                                      size_t concatenation_axis)
                        : m_args(args)
                        , m_out(out)
                        , m_arg_shapes(arg_shapes)
                        , m_out_shape(out_shape)
                        , m_concatenation_axis(concatenation_axis)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        std::vector<typename ET::type*> args;

                        for (auto arg_tv : m_args)
                        {
                            args.push_back(get_tensor_data_ptr<ET>(call_frame, arg_tv));
                        }

                        typename ET::type* out = get_tensor_data_ptr<ET>(call_frame, m_out);

                        kernel::concat<typename ET::type>(
                            args, out, m_arg_shapes, m_out_shape, m_concatenation_axis);
                    }

                protected:
                    std::vector<TensorViewInfo> m_args;
                    TensorViewInfo m_out;
                    std::vector<Shape> m_arg_shapes;
                    Shape m_out_shape;
                    size_t m_concatenation_axis;
                };
            }
        }
    }
}
