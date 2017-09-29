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
            // Intended substitutions for T are shared_ptr<ParameterizedTensorView<...>>
            // and ParameterizedTensorView<...>*.
            template <typename T>
            void concat_matrix(std::vector<T>& args, T out, size_t axis)
            {
                auto mat_out = get_map_matrix_2d(&*out);
                auto& out_shape = out->get_shape();

                assert (out_shape.size() == 2);
                assert (axis == 0 || axis == 1);

                size_t concat_pos = 0;

                for(T arg : args)
                {
                    auto mat_arg = get_map_matrix_2d(&*arg);
                    auto& arg_shape = arg->get_shape();

                    assert (arg_shape.size() == 2);

                    if (axis == 0)
                    {
                        mat_out.block(concat_pos,0,arg_shape.at(0),arg_shape.at(1))
                          << mat_arg;
                        concat_pos += arg_shape.at(0);
                    }
                    else
                    {
                        mat_out.block(0,concat_pos,arg_shape.at(0),arg_shape.at(1))
                          << mat_arg;
                        concat_pos += arg_shape.at(1);
                    }
                }
            }

            template <typename ET>
            class ConcatMatrixInstruction : public Instruction
            {
            public:
                ConcatMatrixInstruction(const std::vector<TensorViewInfo>& args, size_t axis, size_t out)
                    : m_args(args)
                    , m_axis(axis)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    std::vector<ParameterizedTensorView<ET>*> ptvs;
                    for(auto arg : m_args)
                    {
                        ptvs.push_back(call_frame.get_parameterized_tensor_view<ET>(arg.index));
                    }
                    runtime::eigen::concat_matrix(
                        ptvs,
                        call_frame.get_parameterized_tensor_view<ET>(m_out),
                        m_axis);
                }

            protected:
                std::vector<TensorViewInfo> m_args;
                size_t m_axis;
                size_t m_out;
            };
        }
    }
}
