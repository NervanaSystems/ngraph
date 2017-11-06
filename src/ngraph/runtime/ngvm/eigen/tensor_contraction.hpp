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

#include <array>

#include "ngraph/runtime/ngvm/call_frame.hpp"
#include "ngraph/runtime/ngvm/eigen/utils.hpp"
#include "ngraph/runtime/ngvm/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace ngvm
        {
            namespace eigen
            {
                template <typename ET, size_t RANK0, size_t RANK1>
                class TensorContractionInstruction : public Instruction
                {
                public:
                    TensorContractionInstruction(const TensorViewInfo& arg0,
                                                 const TensorViewInfo& arg1,
                                                 size_t arg0_axis,
                                                 size_t arg1_axis,
                                                 const TensorViewInfo& out)
                        : m_arg0(arg0)
                        , m_arg1(arg1)
                        , m_arg0_axis(arg0_axis)
                        , m_arg1_axis(arg1_axis)
                        , m_out(out)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        std::array<Eigen::IndexPair<size_t>, 1> axis_pairs{
                            {Eigen::IndexPair<size_t>(m_arg0_axis, m_arg1_axis)}};
                        EigenTensor<ET, RANK0 + RANK1 - 2>(call_frame, m_out).as_base() =
                            EigenTensor<ET, RANK0>(call_frame, m_arg0)
                                .as_base()
                                .contract(EigenTensor<ET, RANK1>(call_frame, m_arg1).as_base(),
                                          axis_pairs);
                    }

                protected:
                    TensorViewInfo m_arg0;
                    TensorViewInfo m_arg1;
                    size_t m_arg0_axis;
                    size_t m_arg1_axis;
                    TensorViewInfo m_out;
                };
            }
        }
    }
}
