/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>

#include "ngraph/runtime/cpu/cpu_call_frame.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/cpu_tracing.hpp"

using namespace std;
using namespace ngraph;

runtime::cpu::CPU_CallFrame::CPU_CallFrame(std::shared_ptr<CPU_ExternalFunction> external_function,
                                           EntryPoint compiled_function)
    : m_external_function(external_function)
    , m_compiled_function(compiled_function)
{
    setup_runtime_context();
}

runtime::cpu::CPU_CallFrame::~CPU_CallFrame()
{
    cleanup_runtime_context();
}

void runtime::cpu::CPU_CallFrame::call(
    const std::vector<std::shared_ptr<runtime::TensorView>>& output_tvs,
    const std::vector<std::shared_ptr<runtime::TensorView>>& input_tvs)
{
    vector<void*> inputs;
    vector<void*> outputs;

    propagate_layouts(input_tvs, m_external_function->get_parameter_layout_descriptors());
    propagate_layouts(output_tvs, m_external_function->get_result_layout_descriptors());

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensorView> tv =
            static_pointer_cast<runtime::cpu::CPUTensorView>(input_tvs[i]);
        inputs.push_back(tv->get_data_ptr());
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::cpu::CPUTensorView> tv =
            static_pointer_cast<runtime::cpu::CPUTensorView>(output_tvs[i]);
        outputs.push_back(tv->get_data_ptr());
    }

    // Invoke compiled computation
    m_compiled_function(inputs.data(), outputs.data(), ctx);

    if (runtime::cpu::IsTracingEnabled())
    {
        GenerateTimeline(m_external_function->get_op_attrs(),
                         ctx->op_durations,
                         m_external_function->get_function_name() + ".timeline.json");
    }
}

void runtime::cpu::CPU_CallFrame::propagate_layouts(
    const std::vector<std::shared_ptr<runtime::TensorView>>& tvs,
    const LayoutDescriptorPtrs& layouts) const
{
    if (layouts.size() != tvs.size())
    {
        throw ngraph_error(
            "Error propagating layouts - tensor view and layout descriptor counts do not match");
    }
    for (size_t i = 0; i < tvs.size(); i++)
    {
        if (layouts[i] == nullptr)
        {
            throw ngraph_error(
                "Error propagating layouts - layout information missing from tensor view");
        }
        tvs[i]->get_descriptor()->set_tensor_view_layout(layouts[i]);
    }
}

void runtime::cpu::CPU_CallFrame::setup_runtime_context()
{
    ctx = new CPURuntimeContext;

    ctx->op_durations = nullptr;
    if (runtime::cpu::IsTracingEnabled())
    {
        ctx->op_durations = new int64_t[m_external_function->get_op_attrs().size()];
    }
    const auto& mkldnn_emitter = m_external_function->get_mkldnn_emitter();
    ctx->mkldnn_primitives = mkldnn_emitter->get_mkldnn_primitives().data();
    ctx->mkldnn_workspaces = mkldnn_emitter->get_mkldnn_workspaces().data();
}

void runtime::cpu::CPU_CallFrame::cleanup_runtime_context()
{
    delete[] ctx->op_durations;
    delete ctx;
}
