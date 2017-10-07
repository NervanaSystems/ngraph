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

#include "ngraph/runtime/ngvm/call_frame.hpp"
#include "ngraph/runtime/ngvm/eigen/utils.hpp"
#include "ngraph/runtime/ngvm/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"
#include "ngraph/runtime/tensor_view_info.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ngvm
        {
            namespace eigen
            {
                // Would be better to just generate a sequence of copy into slice of output instructions
                template <typename ET>
                class ConcatVectorInstruction : public Instruction
                {
                public:
                    ConcatVectorInstruction(const std::vector<TensorViewInfo>& args,
                                            const TensorViewInfo& out)
                        : m_args(args)
                        , m_out(out)
                    {
                        for (auto arg : args)
                        {
                            auto& arg_shape = arg.get_tensor_view_layout()->get_shape();
                            m_sizes.push_back(arg_shape.at(0));
                        }
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        EigenVector<ET> out(call_frame, m_out);
                        size_t concat_pos = 0;
                        for (size_t i = 0; i < m_args.size(); i++)
                        {
                            out.segment(concat_pos, m_sizes[i])
                                << EigenVector<ET>(call_frame, m_args.at(i));
                            concat_pos += m_sizes[i];
                        }
                    }

                protected:
                    std::vector<TensorViewInfo> m_args;
                    TensorViewInfo m_out;
                    std::vector<size_t> m_sizes;
                };
            }
        }
    }
}
