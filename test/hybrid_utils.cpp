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

#include "hybrid_utils.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/pass/assign_placement.hpp"
#include "ngraph/pass/manager.hpp"

using namespace std;
using namespace ngraph;

TestBackend::TestBackend(const vector<shared_ptr<runtime::Backend>>& backend_list)
    : m_backend_list{backend_list}
{
    if (m_backend_list.size() == 0)
    {
        throw runtime_error("TestBackend backend list empty");
    }
}

shared_ptr<runtime::Tensor> TestBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape)
{
    return m_backend_list[0]->create_tensor(element_type, shape);
}

shared_ptr<runtime::Tensor> TestBackend::create_tensor(const element::Type& element_type,
                                                       const Shape& shape,
                                                       void* memory_pointer)
{
    return m_backend_list[0]->create_tensor(element_type, shape, memory_pointer);
}

bool TestBackend::compile(shared_ptr<Function> func)
{
    if (m_function_map.find(func) == m_function_map.end())
    {
        // Clone function
        FunctionInstance instance;
        instance.m_function = clone_function(*func);

        // Run placement pass
        pass::Manager pass_manager;
        pass_manager.register_pass<pass::AssignPlacement>(m_backend_list);
        pass_manager.run_passes(instance.m_function);

        // Split function to sub_functions
        tie(instance.m_sub_functions, instance.m_map_parameter_to_result) =
            split_function_by_placement_size(instance.m_function);
        m_function_map.insert({func, instance});

        // Compile subfunctions in corresponding backends
        for (shared_ptr<Function>& sub_function : instance.m_sub_functions)
        {
            size_t placement = get_colocated_function_placement_size(sub_function);
            auto backend =
                m_backend_list[(placement - 1)]; // (placement-1) as 0 is default placement
            backend->compile(sub_function);
        }
    }

    return true;
}

bool TestBackend::call(shared_ptr<Function> func,
                       const vector<shared_ptr<runtime::Tensor>>& outputs,
                       const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    // Get FunctionInstance
    bool rc = true;
    auto it = m_function_map.find(func);
    if (it == m_function_map.end())
    {
        compile(func);
        it = m_function_map.find(func);
    }

    throw runtime_error("TestBackend call not supported");

    // for (auto backend : m_backend_list)
    // {
    //     if (backend->is_supported(node))
    //     {
    //         // backend supports the op
    //     }
    // }
    // return true;
}

BackendWrapper::BackendWrapper(const string& backend_name,
                               const set<string>& supported_ops,
                               const string& name)
    : m_backend{runtime::Backend::create(backend_name)}
    , m_supported_ops{supported_ops}
    , m_name{name}
{
}

shared_ptr<runtime::Tensor> BackendWrapper::create_tensor(const element::Type& element_type,
                                                          const Shape& shape)
{
    return m_backend->create_tensor(element_type, shape);
}

shared_ptr<runtime::Tensor> BackendWrapper::create_tensor(const element::Type& element_type,
                                                          const Shape& shape,
                                                          void* memory_pointer)
{
    return m_backend->create_tensor(element_type, shape, memory_pointer);
}

bool BackendWrapper::compile(shared_ptr<Function> func)
{
    return m_backend->compile(func);
}

bool BackendWrapper::call(shared_ptr<Function> func,
                          const vector<shared_ptr<runtime::Tensor>>& outputs,
                          const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    return m_backend->call(func, outputs, inputs);
}

bool BackendWrapper::is_supported(std::shared_ptr<Node> node) const
{
    return m_supported_ops.find(node->description()) != m_supported_ops.end();
}
