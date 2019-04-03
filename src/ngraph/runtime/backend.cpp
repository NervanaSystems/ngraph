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

#include <sstream>

#include "ngraph/file_util.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

runtime::Backend::~Backend()
{
}

std::shared_ptr<ngraph::Node> runtime::Backend::get_backend_op(const std::string& op_name, ...)
{
    std::shared_ptr<ngraph::Node> dummy_node(nullptr);
    return dummy_node;
}

unique_ptr<runtime::Backend> runtime::Backend::create(const string& type)
{
    return BackendManager::create_backend(type);
}

vector<string> runtime::Backend::get_registered_devices()
{
    return BackendManager::get_registered_backends();
}

std::shared_ptr<runtime::Executable>
    runtime::Backend::compile(std::shared_ptr<Function> func,
                              ngraph::pass::PassConfig& pass_config,
                              bool enable_performance_data)
{
    return compile(func, enable_performance_data);
}

bool runtime::Backend::is_supported(const Node& node) const
{
    // The default behavior is that a backend does not support any ops. If this is not the case
    // then override this method and enhance.
    return false;
}

bool runtime::Backend::is_supported_property(const Property prop) const
{
    return false;
}

void runtime::Backend::remove_compiled_function(std::shared_ptr<Executable> exec)
{
}

runtime::AllocateFunc runtime::Backend::get_device_memory_alloc()
{
    // override this method from all supported backends to return memory allocator
    // which allocates device pinned memory
    return nullptr;
}

runtime::DestroyFunc runtime::Backend::get_device_memory_dealloc()
{
    // override this method from all supported backends to return memory de-allocator
    // which de-allocates device pinned memory
    return nullptr;
}

bool runtime::Backend::is_device_memory(void* ptr)
{
    // override this method for each supported backend to determine if the passed pointer is in
    // device pinned memory or not
    return false;
}
