//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <algorithm>

#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/cpu/cpu_debugger.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_Debugger::CPU_Debugger(ngraph::runtime::cpu::CPU_CallFrame& callframe)
    : m_callframe(callframe)
{
}

runtime::cpu::CPU_Debugger::~CPU_Debugger()
{
}

bool runtime::cpu::CPU_Debugger::step()
{
    auto ctx = m_callframe.ctx;
    if (ctx->pc >= m_callframe.m_external_function->op_names.size())
    {
        return false;
    }

    bool is_set = ctx->breakpoints.count(ctx->pc + 1) != 0;
    ctx->breakpoints.insert(ctx->pc + 1);
    m_callframe.inner_call(m_outputs, m_inputs);
    if (!is_set)
    {
        ctx->breakpoints.erase(ctx->pc);
    }
    return true;
}

void runtime::cpu::CPU_Debugger::resume()
{
    auto ctx = m_callframe.ctx;
    if (ctx->pc >= m_callframe.m_external_function->op_names.size())
    {
        return;
    }

    m_callframe.inner_call(m_outputs, m_inputs);
    return;
}

void runtime::cpu::CPU_Debugger::call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    m_outputs.assign(outputs.begin(), outputs.end());
    m_inputs.assign(inputs.begin(), inputs.end());
    m_callframe.ctx->pc = 0;
    m_callframe.inner_call(m_outputs, m_inputs);
}

bool runtime::cpu::CPU_Debugger::add_breakpoint(std::shared_ptr<Node> op)
{
    auto external_function = m_callframe.m_external_function;
    auto ctx = m_callframe.ctx;
    auto i_pos = std::find(
        external_function->op_names.begin(), external_function->op_names.end(), op->get_name());
    if (i_pos != external_function->op_names.end())
    {
        auto pc = static_cast<size_t>(std::distance(external_function->op_names.begin(), i_pos));
        ctx->breakpoints.insert(pc);
        return true;
    }
    return false;
}

bool runtime::cpu::CPU_Debugger::delete_breakpoint(std::shared_ptr<Node> op)
{
    auto external_function = m_callframe.m_external_function;
    auto ctx = m_callframe.ctx;
    auto i_pos = std::find(
        external_function->op_names.begin(), external_function->op_names.end(), op->get_name());
    if (i_pos != external_function->op_names.end())
    {
        auto pc = static_cast<size_t>(std::distance(external_function->op_names.begin(), i_pos));
        ctx->breakpoints.erase(pc);
        return true;
    }
    return false;
}

void* runtime::cpu::CPU_Debugger::inspect(std::shared_ptr<Node> op, size_t output_index)
{
    return m_callframe.m_external_function->tensor_data.at(op->get_name() + "_" +
                                                           to_string(output_index));
}
