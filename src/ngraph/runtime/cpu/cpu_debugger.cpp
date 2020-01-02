//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"

using namespace std;
using namespace ngraph;

void runtime::cpu::CPU_CountTracepoint::operator()(void** outputs, const std::string& name)
{
    if (m_count == 0)
    {
        return;
    }
    if (++m_iteration >= m_count)
    {
        m_callback(outputs, name);
        m_iteration = 0;
    }
}

runtime::cpu::CPU_Debugger::CPU_Debugger(ngraph::runtime::cpu::CPU_CallFrame& callframe)
    : m_callframe(callframe)
{
}

runtime::cpu::CPU_Debugger::~CPU_Debugger()
{
}

bool runtime::cpu::CPU_Debugger::step()
{
    auto ctx = m_callframe.m_ctx_vec[0];
    if (ctx->pc >= m_callframe.m_external_function->op_names.size())
    {
        return false;
    }

    bool is_set = ctx->breakpoints.count(ctx->pc + 1) != 0;
    ctx->breakpoints.insert(ctx->pc + 1);
    m_callframe.inner_call(m_outputs, m_inputs, 0);
    if (!is_set)
    {
        ctx->breakpoints.erase(ctx->pc);
    }
    return true;
}

void runtime::cpu::CPU_Debugger::resume()
{
    auto ctx = m_callframe.m_ctx_vec[0];
    if (ctx->pc >= m_callframe.m_external_function->op_names.size())
    {
        return;
    }

    m_callframe.inner_call(m_outputs, m_inputs, 0);
    return;
}

void runtime::cpu::CPU_Debugger::call(const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
                                      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs)
{
    m_outputs.assign(outputs.begin(), outputs.end());
    m_inputs.assign(inputs.begin(), inputs.end());
    m_callframe.m_ctx_vec[0]->pc = 0;
    m_callframe.inner_call(m_outputs, m_inputs, 0);
}

std::tuple<bool, size_t> runtime::cpu::CPU_Debugger::find_pc_for_node(std::shared_ptr<Node> op)
{
    auto external_function = m_callframe.m_external_function;
    auto i_pos = std::find(
        external_function->op_names.begin(), external_function->op_names.end(), op->get_name());

    if (i_pos != external_function->op_names.end())
    {
        auto pc = static_cast<size_t>(std::distance(external_function->op_names.begin(), i_pos));
        return std::tuple<bool, size_t>{true, pc};
    }
    return std::tuple<bool, size_t>{false, 0};
}

bool runtime::cpu::CPU_Debugger::add_breakpoint(std::shared_ptr<Node> op)
{
    bool found;
    size_t pc;
    std::tie(found, pc) = find_pc_for_node(op);
    if (found)
    {
        m_callframe.m_ctx_vec[0]->breakpoints.insert(pc);
        return true;
    }
    return false;
}

bool runtime::cpu::CPU_Debugger::delete_breakpoint(std::shared_ptr<Node> op)
{
    bool found;
    size_t pc;
    std::tie(found, pc) = find_pc_for_node(op);
    if (found)
    {
        m_callframe.m_ctx_vec[0]->breakpoints.erase(pc);
        return true;
    }
    return false;
}

void* runtime::cpu::CPU_Debugger::inspect(std::shared_ptr<Node> op, size_t output_index)
{
    if (m_callframe.m_external_function->is_direct_execution())
    {
        auto index = m_callframe.m_external_function->get_buffer_index(op->get_name() + "_" +
                                                                       to_string(output_index));
        return m_callframe.m_ctx_vec[0]->buffer_data[index];
    }
    else
    {
        auto index = m_callframe.m_external_function->m_buffer_indices.at(op->get_name() + "_" +
                                                                          to_string(output_index));
        return m_callframe.m_ctx_vec[0]->buffer_data[index];
    }
}

bool runtime::cpu::CPU_Debugger::add_tracepoint(
    std::shared_ptr<Node> op, const std::function<void(void**, const std::string&)>& callback)
{
    auto external_function = m_callframe.m_external_function;
    bool found;
    size_t pc;
    std::tie(found, pc) = find_pc_for_node(op);
    if (found)
    {
        if (replaced_functors.count(pc) != 0)
        {
            return false;
        }

        auto op_name = op->get_name();
        std::vector<size_t> poutputs;
        for (size_t i = 0; i < op->get_outputs().size(); i++)
        {
            poutputs.push_back(external_function->get_buffer_index(op_name + "_" + to_string(i)));
        }

        auto original_functor = external_function->functors.at(pc);

        auto trace_functor = [poutputs, callback, original_functor, op_name](
            CPURuntimeContext* ctx, CPUExecutionContext* ectx) {

            original_functor(ctx, ectx);

            std::vector<void*> outputs;
            for (auto pout : poutputs)
            {
                outputs.push_back(ctx->buffer_data[pout]);
            }

            callback(outputs.data(), op_name);
        };
        replaced_functors[pc] = original_functor;
        external_function->functors.at(pc) = trace_functor;
        return true;
    }

    return false;
}

bool runtime::cpu::CPU_Debugger::delete_tracepoint(std::shared_ptr<Node> op)
{
    bool found;
    size_t pc;
    std::tie(found, pc) = find_pc_for_node(op);
    if (found)
    {
        m_callframe.m_external_function->functors.at(pc) = replaced_functors.at(pc);
        return true;
    }

    return false;
}
