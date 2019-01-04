//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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

#include "ngraph/runtime/hybrid/hybrid_backend.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_tensor.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/hybrid/hybrid_executable.hpp"
#include "ngraph/runtime/hybrid/hybrid_util.hpp"
#include "ngraph/runtime/hybrid/pass/assign_placement.hpp"
#include "ngraph/runtime/hybrid/pass/fix_get_output_element.hpp"
#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/runtime/tensor.hpp"

using namespace ngraph;
using namespace std;

runtime::hybrid::HybridBackend::HybridBackend(
    const std::vector<std::shared_ptr<runtime::Backend>>& backend_list)
    : m_backend_list{backend_list}
{
    NGRAPH_INFO;
}

shared_ptr<runtime::Tensor>
    runtime::hybrid::HybridBackend::create_tensor(const element::Type& element_type,
                                                  const Shape& shape)
{
    auto it = m_backend_list.begin();
    return (*it)->create_tensor(element_type, shape);
}

shared_ptr<runtime::Tensor> runtime::hybrid::HybridBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    auto it = m_backend_list.begin();
    return (*it)->create_tensor(element_type, shape, memory_pointer);
}

unique_ptr<runtime::Executable>
    runtime::hybrid::HybridBackend::compile(shared_ptr<Function> func,
                                            bool enable_performance_collection)
{
    std::unique_ptr<HybridExecutable> exec{
        new HybridExecutable(m_backend_list, func, enable_performance_collection)};

    return exec;
}

bool runtime::hybrid::HybridBackend::is_supported(const Node& node) const
{
    return true;
}

string runtime::hybrid::HybridBackend::get_placement_name(const runtime::Tensor* t)
{
    string rc;
    if (dynamic_cast<const runtime::HostTensor*>(t) != nullptr)
    {
        rc = "HostTensor";
    }
    else if (dynamic_cast<const runtime::gpu::GPUTensor*>(t) != nullptr)
    {
        rc = "GPUTensor";
    }
    return rc;
}
string runtime::hybrid::HybridBackend::get_placement_name(const runtime::Backend* t)
{
    string rc;
    if (dynamic_cast<const runtime::interpreter::INTBackend*>(t) != nullptr)
    {
        rc = "INTBackend";
    }
    else if (dynamic_cast<const runtime::gpu::GPU_Backend*>(t) != nullptr)
    {
        rc = "GPU_Backend";
    }
    return rc;
}
size_t runtime::hybrid::HybridBackend::get_placement(const runtime::Tensor* t)
{
    size_t index = 0;
    for (const shared_ptr<ngraph::runtime::Backend>& be : m_backend_list)
    {
        if (t->get_parent() == be.get())
        {
            return index;
        }
        index++;
    }
    return -1;
}
