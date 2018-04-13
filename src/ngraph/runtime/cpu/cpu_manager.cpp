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

#include <memory>

#include "ngraph/runtime/cpu/cpu_backend.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_manager.hpp"

using namespace std;
using namespace ngraph;

std::shared_ptr<ngraph::runtime::Backend> runtime::cpu::CPU_Manager::allocate_backend()
{
    return std::make_shared<CPU_Backend>();
}

vector<size_t> runtime::cpu::CPU_Manager::get_subdevices() const
{
    throw runtime_error("unimplemented");
}

ngraph::runtime::Manager::Factory runtime::cpu::CPU_Manager::factory =
    ngraph::runtime::Manager::register_factory(
        "CPU", [](const std::string& name) -> std::shared_ptr<ngraph::runtime::Manager> {
            return std::make_shared<CPU_Manager>();
        });
