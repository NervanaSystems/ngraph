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

#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"

#include "ngraph/runtime/gpu/gpu_call_frame.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
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
    // Host tensors
    vector<void*> inputs;
    vector<void*> outputs;

    for (size_t i = 0; i < input_tvs.size(); i++)
    {
        shared_ptr<runtime::HostTensorView> tv =
            static_pointer_cast<runtime::HostTensorView>(input_tvs[i]);
        inputs.push_back(tv->get_data_ptr());
    }
    for (size_t i = 0; i < output_tvs.size(); i++)
    {
        shared_ptr<runtime::HostTensorView> tv =
            static_pointer_cast<runtime::HostTensorView>(output_tvs[i]);
        outputs.push_back(tv->get_data_ptr());
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
