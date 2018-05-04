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

#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

static bool static_init()
{
    runtime::Backend::register_backend("GPU", make_shared<runtime::gpu::GPU_Backend>());
    return true;
};

bool runtime::gpu::GPU_Backend::init = static_init();

shared_ptr<runtime::gpu::GPU_CallFrame> runtime::gpu::GPU_Backend::make_call_frame(
    const shared_ptr<GPU_ExternalFunction>& external_function)
{
    return external_function->make_call_frame();
}

shared_ptr<runtime::TensorView>
    runtime::gpu::GPU_Backend::create_tensor(const element::Type& element_type, const Shape& shape)
{
    return make_shared<runtime::gpu::GPU_TensorView>(element_type, shape);
}

shared_ptr<runtime::TensorView> runtime::gpu::GPU_Backend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::gpu::GPU_TensorView>(element_type, shape, memory_pointer);
}

bool runtime::gpu::GPU_Backend::compile(shared_ptr<Function> func)
{
    if (!contains_key(m_function_map, func))
    {
        FunctionInstance instance;
        instance.m_external_function = make_shared<GPU_ExternalFunction>(func);
        auto cf = instance.m_external_function->make_call_frame();
        instance.m_call_frame = dynamic_pointer_cast<GPU_CallFrame>(cf);
        m_function_map.insert({func, instance});
    }
    return true;
}

bool runtime::gpu::GPU_Backend::call(shared_ptr<Function> func,
                                     const vector<shared_ptr<runtime::TensorView>>& outputs,
                                     const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    bool rc = true;

    validate_call(func, outputs, inputs);

    auto it = m_function_map.find(func);
    if (it == m_function_map.end())
    {
        compile(func);
        it = m_function_map.find(func);
    }

    if (it == m_function_map.end())
    {
        throw runtime_error("Error constructing backend.");
    }

    FunctionInstance& instance = it->second;
    instance.m_call_frame->call(outputs, inputs);

    return rc;
}
