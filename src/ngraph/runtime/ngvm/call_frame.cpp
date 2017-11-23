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

#include <algorithm>

#include "ngraph/runtime/ngvm/call_frame.hpp"
#include "ngraph/runtime/ngvm/instruction.hpp"

using namespace std;
using namespace ngraph::runtime::ngvm;

CallFrame::CallFrame(size_t n_inputs,
                     size_t n_outputs,
                     size_t frame_size,
                     const TensorViewPtrs& temps,
                     size_t initial_pc,
                     const shared_ptr<vector<shared_ptr<Instruction>>>& instructions)

    : m_n_inputs(n_inputs)
    , m_n_outputs(n_outputs)
    , m_frame_size(frame_size)
    , m_tensor_views(m_frame_size)
    , m_initial_pc(initial_pc)
    , m_instructions(instructions)
{
    copy(temps.begin(), temps.end(), m_tensor_views.begin() + m_n_inputs + m_n_outputs);
}

void CallFrame::tensor_call(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& inputs,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& outputs)
{
    if (inputs.size() != m_n_inputs)
    {
        throw ngraph_error("Incorrect number of inputs");
    }
    copy(inputs.begin(), inputs.end(), m_tensor_views.begin());
    if (outputs.size() != m_n_outputs)
    {
        throw ngraph_error("Incorrect number of outputs");
    }
    copy(outputs.begin(), outputs.end(), m_tensor_views.begin() + m_n_inputs);
    m_next_pc = m_initial_pc;
    m_return = false;
    while (!m_return)
    {
        m_pc = m_next_pc;
        m_next_pc = m_pc + 1;
        m_instructions->at(m_pc)->execute(*this);
    }
    // Don't hold onto inputs/outputs
    fill_n(m_tensor_views.begin(), m_n_inputs + m_n_outputs, nullptr);
}

void CallFrame::call(const std::vector<std::shared_ptr<ngraph::runtime::Value>>& arguments,
                     const std::vector<std::shared_ptr<ngraph::runtime::Value>>& results)
{
    // TODO: Check types of args and result
    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> inputs;
    for (auto argument : arguments)
    {
        argument->collect_tensor_views(inputs, argument);
    }

    std::vector<std::shared_ptr<ngraph::runtime::TensorView>> outputs;
    for (auto result : results)
    {
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}
