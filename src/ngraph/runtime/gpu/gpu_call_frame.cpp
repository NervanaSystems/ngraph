// ----------------------------------------------------------------------------
// copyright 2017 nervana systems inc.
// licensed under the apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// you may obtain a copy of the license at
//
//      http://www.apache.org/licenses/license-2.0
//
// unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// see the license for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <fstream>

#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view.hpp"

using namespace std;
using namespace ngraph;

runtime::gpu::GPU_CallFrame::GPU_CallFrame(std::shared_ptr<GPU_ExternalFunction> external_function,
                                           EntryPoint compiled_function)
    : m_external_function(external_function)
    , m_compiled_function(compiled_function)
{
}

void runtime::gpu::GPU_CallFrame::tensor_call(
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& input_tvs,
    const std::vector<std::shared_ptr<ngraph::runtime::TensorView>>& output_tvs)
{
    vector<void*> inputs;
    vector<void*> outputs;
    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::gpu::GPU_TensorView> tv =
            static_pointer_cast<runtime::gpu::GPU_TensorView>(input_tvs[i]);
        // inputs.push_back(tv->get_data_ptr());
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::gpu::GPU_TensorView> tv =
            static_pointer_cast<runtime::gpu::GPU_TensorView>(output_tvs[i]);
        // outputs.push_back(tv->get_data_ptr());
    }

    // Invoke compiled computation
    m_compiled_function(inputs.data(), outputs.data());
}

void runtime::gpu::GPU_CallFrame::call(
    const std::vector<std::shared_ptr<runtime::TensorView>>& arguments,
    const std::vector<std::shared_ptr<runtime::TensorView>>& results)
{
    // TODO: Check types of args and result
    vector<shared_ptr<runtime::TensorView>> inputs;
    for (shared_ptr<runtime::TensorView> argument : arguments)
    {
        argument->collect_tensor_views(inputs, argument);
    }

    vector<shared_ptr<runtime::TensorView>> outputs;
    for (shared_ptr<runtime::TensorView> result : results)
    {
        result->collect_tensor_views(outputs, result);
    }

    tensor_call(inputs, outputs);
}

vector<runtime::PerformanceCounter> runtime::gpu::GPU_CallFrame::get_performance_data() const
{
    vector<runtime::PerformanceCounter> rc;
    // auto* engine = m_external_function->m_execution_engine.get();
    // if (engine)
    // {
    //     auto get_count = engine->find_function<size_t()>("get_debug_timer_count");
    //     auto get_name = engine->find_function<const char*(size_t)>("get_debug_timer_name");
    //     auto get_microseconds =
    //         engine->find_function<size_t(size_t)>("get_debug_timer_microseconds");
    //     auto get_call_count = engine->find_function<size_t(size_t)>("get_debug_timer_call_count");

    //     if (get_count && get_name && get_microseconds && get_call_count)
    //     {
    //         size_t count = get_count();
    //         for (size_t i = 0; i < count; i++)
    //         {
    //             rc.push_back({get_name(i), get_microseconds(i), get_call_count(i)});
    //         }
    //     }
    // }
    return rc;
}
