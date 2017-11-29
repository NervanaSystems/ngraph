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

#include <memory>

#include "ngraph/runtime/interpreter/call_frame.hpp"
#include "ngraph/runtime/interpreter/kernel/utils.hpp"
#include "ngraph/runtime/interpreter/external_function.hpp"
#include "ngraph/runtime/interpreter/instruction.hpp"
#include "ngraph/runtime/tensor_view.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            namespace kernel
            {
                class CallInstruction : public Instruction
                {
                public:
                    CallInstruction(std::shared_ptr<ExternalFunction> ef,
                                    std::vector<TensorViewInfo> in,
                                    std::vector<TensorViewInfo> out)
                        : m_external_function(ef)
                        , m_in(in)
                        , m_out(out)
                    {
                    }

                    virtual void execute(CallFrame& call_frame) const override
                    {
                        std::shared_ptr<CallFrame> cf = std::dynamic_pointer_cast<CallFrame>(
                            m_external_function->make_call_frame());

                        std::vector<std::shared_ptr<ngraph::runtime::Value>> inputs;
                        std::vector<std::shared_ptr<ngraph::runtime::Value>> outputs;

                        for (auto in : m_in)
                        {
                            inputs.push_back(call_frame.get_tensor_view(in.get_index()));
                        }
                        for (auto out : m_out)
                        {
                            outputs.push_back(call_frame.get_tensor_view(out.get_index()));
                        }
                        cf->call(inputs, outputs);
                    }

                protected:
                    std::shared_ptr<ExternalFunction> m_external_function;
                    std::vector<TensorViewInfo> m_in;
                    std::vector<TensorViewInfo> m_out;
                };
            }
        }
    }
}
