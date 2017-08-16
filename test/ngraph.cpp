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

#include <vector>
#include <string>
#include <sstream>
#include <memory>
#include <dlfcn.h>

#include "gtest/gtest.h"
#include "ngraph.hpp"
#include "log.hpp"

using namespace std;

TEST(NGraph, loadTest)
{
    // load the triangle library
    void* pluginLib = dlopen("../src/libngraph.so", RTLD_LAZY);
    if (!pluginLib)
    {
        std::cerr << "Cannot load library: " << dlerror() << '\n';
        ASSERT_FALSE(true);
    }

    // reset errors
    dlerror();

    // Get the symbols
    auto createPfn = reinterpret_cast<CreatePluginPfn>(dlsym(pluginLib, "create_plugin"));
    ASSERT_FALSE(createPfn == nullptr);

    auto destroyPfn = reinterpret_cast<DestroyPluginPfn>(dlsym(pluginLib, "destroy_plugin"));
    ASSERT_FALSE(destroyPfn == nullptr);

    NGraph* pluginObj = createPfn();

    INFO << "Call a method on the Object";
    ASSERT_EQ("NGraph Plugin", pluginObj->get_name());
    INFO << "Plugin Name: " << pluginObj->get_name();

    // Add some parameters
    const vector<string> TEST_PARAMS = {"param-1", "param-2", "param-3"};

    pluginObj->add_params( TEST_PARAMS );

    // Get the list of params
    auto& storedParams = pluginObj->get_params();
    EXPECT_EQ( TEST_PARAMS.size(), storedParams.size() );
    for (int i = 0; i < TEST_PARAMS.size(); i++)
    {
        EXPECT_EQ( TEST_PARAMS[i], storedParams[i] );
    }

    INFO << "Destroy the Plugin Object";
    destroyPfn(pluginObj);

    dlclose(pluginLib);
}
