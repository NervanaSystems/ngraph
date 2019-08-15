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

#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

#include "ngraph/op/util/gen_op.hpp"

using namespace std;
using namespace ngraph;

std::ostream& op::util::GenOp::write_long_description(std::ostream& str) const
{
    Node::write_long_description(str);
    auto attr_keys = get_attribute_keys();
    if (attr_keys.size() > 0)
    {
        str << "<";
        bool first = true;
        for (auto& k : attr_keys)
        {
            if (!first)
            {
                str << ", ";
            }
            str << k << "=" << get_attribute(k);
            first = false;
        }
        str << ">";
    }
    return str;
}

using BuilderMap = unordered_map<string, GenOpBuilder*>;

static mutex global_builder_map_mutex;

static BuilderMap& global_builder_map()
{
    static BuilderMap* p_map = new BuilderMap();
    return *p_map;
}

bool ngraph::register_gen_op(const char* op_name, GenOpBuilder* op_builder)
{
    lock_guard<mutex> l(global_builder_map_mutex);

    global_builder_map()[string(op_name)] = op_builder;
    return true;
}

const GenOpBuilder* ngraph::get_op_builder(const std::string& op_name)
{
    lock_guard<mutex> l(global_builder_map_mutex);

    if (global_builder_map().count(op_name) == 0)
    {
        return nullptr;
    }
    else
    {
        return global_builder_map()[op_name];
    }
}
