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

#include "ngraph/runtime/opv/opv_backend_visibility.hpp"
#include "ngraph/component_manager.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/except.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/opv/opv_backend.hpp"
#include "ngraph/runtime/opv/opv_executable.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

extern "C" OPV_BACKEND_API void ngraph_register_opv_backend()
{
    runtime::BackendManager::register_backend("OPV", [](const std::string& /* config */) {
        return std::make_shared<runtime::opv::OPVBackend>();
    });
}

runtime::opv::OPVBackend::OPVBackend()
{
}

runtime::opv::OPVBackend::OPVBackend(const vector<string>& unsupported_op_name_list)
    : m_unsupported_op_name_list{unsupported_op_name_list.begin(), unsupported_op_name_list.end()}
{
}

shared_ptr<runtime::Tensor>
    runtime::opv::OPVBackend::create_tensor(const element::Type& type, const Shape& shape)
{
    return make_shared<runtime::opv::OPVTensor>(type, shape);
}

shared_ptr<runtime::Tensor> runtime::opv::OPVBackend::create_tensor(
    const element::Type& type, const Shape& shape, void* memory_pointer)
{
    throw std::runtime_error("create_tensor function is unimplemented for opv backend");
}

shared_ptr<runtime::Executable>
    runtime::opv::OPVBackend::compile(shared_ptr<Function> function,
                                              bool enable_performance_collection)
{
    return make_shared<OPVExecutable>(function, enable_performance_collection);
}

bool runtime::opv::OPVBackend::is_supported(const Node& node) const
{
    return m_unsupported_op_name_list.find(node.description()) == m_unsupported_op_name_list.end();
}

std::shared_ptr<runtime::Executable> runtime::opv::OPVBackend::load(istream& in)
{
    throw std::runtime_error("create_tensor function is unimplemented for opv backend");
}

bool runtime::opv::OPVBackend::set_config(const map<string, string>& config, string& error)
{
    throw std::runtime_error("set_config function is unimplemented for opv backend");
} 