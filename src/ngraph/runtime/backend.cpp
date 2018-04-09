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

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/manager.hpp"

using namespace std;
using namespace ngraph;

std::shared_ptr<runtime::Backend> runtime::Backend::create(const std::string& type)
{
    std::shared_ptr<Manager> manager = runtime::Manager::get(type);
    return manager->allocate_backend();
}

vector<string> runtime::Backend::get_registered_devices()
{
    vector<string> rc;
    for (const pair<string, runtime::Manager::Factory>& p : runtime::Manager::get_factory_map())
    {
        rc.push_back(p.first);
    }
    return rc;
}

vector<size_t> runtime::Backend::get_subdevices(const string& type)
{
    std::shared_ptr<Manager> manager = runtime::Manager::get(type);
    return manager->get_subdevices();
}

void runtime::Backend::remove_compiled_function(const Function& func)
{
}
