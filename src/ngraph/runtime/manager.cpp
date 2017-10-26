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
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <string>

using namespace ngraph::runtime;

bool Manager::load_plugins(const std::string& runtime_plugin_libs)
{
    std::istringstream ss(runtime_plugin_libs);
    std::string plugin_lib_path;

    while (std::getline(ss, plugin_lib_path, ':'))
    {
        if (plugin_lib_path.size() > 0)
        {
            void* lib_handle = dlopen(plugin_lib_path.c_str(), RTLD_NOW);
            if (!lib_handle)
            {
                std::cerr << "Cannot open library: " << plugin_lib_path << ", " << dlerror()
                          << std::endl;
                return false;
            }
            else
            {
                std::cerr << "Loaded runtime at " << lib_handle << std::endl;
            }
        }
    }
    return true;
}

Manager::FactoryMap& Manager::get_factory_map()
{
    // Stores Manager Factories
    static FactoryMap factory_map;

    // Try to load runtime plugins
    if (!Manager::m_is_factory_map_initialized)
    {
        if (!Manager::load_plugins(RUNTIME_PLUGIN_LIBS))
        {
            std::cerr << "Failed to load at least one of the following libraries: "
                      << RUNTIME_PLUGIN_LIBS << std::endl;
        }
        Manager::m_is_factory_map_initialized = true;
    }
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

bool Manager::m_is_factory_map_initialized = false;
