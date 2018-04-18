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

#include "ngraph/runtime/ie/ie_backend.hpp"
#include "ngraph/runtime/ie/ie_manager.hpp"

using namespace ngraph;
using namespace std;

shared_ptr<runtime::Backend> runtime::ie::IE_Manager::allocate_backend()
{
    return make_shared<IE_Backend>();
}

std::vector<size_t> runtime::ie::IE_Manager::get_subdevices() const
{
    vector<size_t> rc;
    return rc;
}

runtime::Manager::Factory runtime::ie::IE_Manager::factory =
    runtime::Manager::register_factory("IE",
                                       [](const string& name) -> shared_ptr<runtime::Manager> {
                                           return make_shared<IE_Manager>();
                                       });
