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

#include "ngraph/runtime/external_function.hpp"
#include "ngraph/runtime/interpreter/call_frame.hpp"
#include "ngraph/runtime/interpreter/eigen/utils.hpp"
#include "ngraph/runtime/interpreter/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace eigen
            {
                template <typename T>
                class ReduceMatrixRowsInstruction : public Instruction
                {
                public:
                    ReduceMatrixRowsInstruction(std::shared_ptr<ExternalFunction> ef,
                                                const TensorViewInfo& arg0,
                                                const TensorViewInfo& arg1,
                                                const TensorViewInfo& out)
                        : m_external_function(ef)
                        , m_arg0(arg0)
                        , m_arg1(arg1)
                        , m_out(out)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        auto ef = m_external_function;
                        auto f = [ef](typename T::type x, typename T::type y) -> typename T::type
                        {
                            std::shared_ptr<CallFrame> cf =
                                std::dynamic_pointer_cast<CallFrame>(ef->make_call_frame());

                            auto tx = ngraph::runtime::make_tensor<T>(Shape{}, {x});
                            auto ty = ngraph::runtime::make_tensor<T>(Shape{}, {y});
                            auto tr = ngraph::runtime::make_tensor<T>(Shape{});

                            cf->call({tx, ty}, {tr});
                            return tr->get_vector()[0];
                        };
                        EigenVector<T>(out) =
                            EigenMatrix<T>(arg0).rowwise().redux(f);
                    }

                protected:
                    std::shared_ptr<ExternalFunction> m_external_function;
                    TensorViewInfo m_arg0;
                    TensorViewInfo m_arg1;
                    TensorViewInfo m_out;
                };
            }
        }
    }
}
