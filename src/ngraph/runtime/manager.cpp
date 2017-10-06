// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/runtime/manager.hpp"

using namespace ngraph::runtime;

Manager::FactoryMap& Manager::get_factory_map()
{
    static FactoryMap factory_map;
    return factory_map;
}

std::shared_ptr<Manager> Manager::get(const std::string& name)
{
    return get_factory_map().at(name)(name);
}

Manager::Factory Manager::register_factory(std::string name, Factory factory)
{
    get_factory_map()[name] = factory;
    return factory;
}
