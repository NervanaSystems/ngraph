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
                template <typename ET>
                class MatrixSliceInstruction : public Instruction
                {
                public:
                    MatrixSliceInstruction(const TensorViewInfo& arg,
                                           const TensorViewInfo& out,
                                           size_t lower_row,
                                           size_t lower_col,
                                           size_t upper_row,
                                           size_t upper_col)
                        : m_arg(arg)
                        , m_out(out)
                        , m_lower_row(lower_row)
                        , m_lower_col(lower_col)
                        , m_upper_row(upper_row)
                        , m_upper_col(upper_col)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        EigenMatrix<ET>(call_frame, m_out) = EigenMatrix<ET>(call_frame, m_arg)
                                                                 .block(m_lower_row,
                                                                        m_lower_col,
                                                                        m_upper_row - m_lower_row,
                                                                        m_upper_col - m_lower_col);
                    }

                protected:
                    TensorViewInfo m_arg;
                    TensorViewInfo m_out;
                    size_t m_lower_row;
                    size_t m_lower_col;
                    size_t m_upper_row;
                    size_t m_upper_col;
                };
            }
        }
    }
}
