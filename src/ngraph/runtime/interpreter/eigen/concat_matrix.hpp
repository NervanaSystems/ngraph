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

#include "ngraph/runtime/interpreter/call_frame.hpp"
#include "ngraph/runtime/interpreter/eigen/utils.hpp"
#include "ngraph/runtime/interpreter/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace eigen
            {
                template <typename T>
                class ConcatMatrixInstruction : public Instruction
                {
                public:
                    ConcatMatrixInstruction(const std::vector<TensorViewInfo>& args,
                                            size_t axis,
                                            const TensorViewInfo& out)
                        : m_args(args)
                        , m_axis(axis)
                        , m_out(out)
                    {
                        size_t concat_pos[2]{0, 0};
                        for (auto arg : args)
                        {
                            auto& arg_shape = arg.get_tensor_view_layout()->get_shape();
                            m_blocks.push_back(
                                {concat_pos[0], concat_pos[1], arg_shape.at(0), arg_shape.at(1)});
                            concat_pos[axis] += arg_shape.at(axis);
                        }
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        EigenMatrix<T> out(out);
                        for (size_t i = 0; i < m_args.size(); i++)
                        {
                            auto& b = m_blocks[i];
                            out.block(b[0], b[1], b[2], b[3])
                                << EigenMatrix<T>(args.at(i));
                        }
                    }

                protected:
                    std::vector<TensorViewInfo> m_args;
                    size_t m_axis;
                    TensorViewInfo m_out;
                    std::vector<std::vector<size_t>> m_blocks;
                };
            }
        }
    }
}
