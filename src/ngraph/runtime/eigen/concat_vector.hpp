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

namespace ngraph
{
    namespace runtime
    {
        namespace eigen
        {
            // Intended substitutions for T are shared_ptr<ParameterizedTensorView<...>>
            // and ParameterizedTensorView<...>*.
            template <typename T>
            void concat_vector(std::vector<T>& args, T out)
            {
                auto vec_out = get_map_matrix(&*out);
                auto& out_shape = out->get_shape();

                assert(out_shape.size() == 1);

                size_t concat_pos = 0;

                for (T arg : args)
                {
                    auto vec_arg = get_map_matrix(&*arg);
                    auto& arg_shape = arg->get_shape();

                    assert(arg_shape.size() == 1);

                    vec_out.segment(concat_pos, arg_shape.at(0)) << vec_arg;
                    concat_pos += arg_shape.at(0);
                }
            }

            template <typename ET>
            class ConcatVectorInstruction : public Instruction
            {
            public:
                ConcatVectorInstruction(const std::vector<size_t>& args, size_t out)
                    : m_args(args)
                    , m_out(out)
                {
                }

                virtual void execute(CallFrame& call_frame) const override
                {
                    std::vector<ParameterizedTensorView<ET>*> ptvs;
                    for (size_t arg : m_args)
                    {
                        ptvs.push_back(call_frame.get_parameterized_tensor_view<ET>(arg));
                    }
                    runtime::eigen::concat_vector(
                        ptvs, call_frame.get_parameterized_tensor_view<ET>(m_out));
                }

            protected:
                std::vector<size_t> m_args;
                size_t m_out;
            };
        }
    }
}
