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

#include "ngraph/runtime/interpreter/int_backend.hpp"
#include "ngraph/except.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/interpreter/int_executable.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

extern "C" runtime::BackendConstructor* get_backend_constructor_pointer()
{
    class INTBackendConstructor : public runtime::BackendConstructor
    {
    public:
        std::shared_ptr<runtime::Backend> create(const std::string& config) override
        {
            return std::make_shared<runtime::interpreter::INTBackend>();
        }
    };

    static unique_ptr<runtime::BackendConstructor> s_backend_constructor(
        new INTBackendConstructor());
    return s_backend_constructor.get();
}

runtime::interpreter::INTBackend::INTBackend()
{
}

runtime::interpreter::INTBackend::INTBackend(const vector<string>& unsupported_op_name_list)
    : m_unsupported_op_name_list{unsupported_op_name_list.begin(), unsupported_op_name_list.end()}
{
}

shared_ptr<runtime::Tensor>
    runtime::interpreter::INTBackend::create_tensor(const element::Type& type, const Shape& shape)
{
    return make_shared<runtime::HostTensor>(type, shape);
}

shared_ptr<runtime::Tensor> runtime::interpreter::INTBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::HostTensor>(type, shape, memory_pointer);
}

shared_ptr<runtime::Executable>
    runtime::interpreter::INTBackend::compile(shared_ptr<Function> function,
                                              bool enable_performance_collection)
{
    return make_shared<INTExecutable>(function, enable_performance_collection);
}

bool runtime::interpreter::INTBackend::is_supported(const Node& node) const
{
    return m_unsupported_op_name_list.find(node.description()) == m_unsupported_op_name_list.end();
}
