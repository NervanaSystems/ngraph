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

#include <memory>

#include "ngraph/builder/xla_tuple.hpp"
#include "ngraph/except.hpp"

using namespace std;
using namespace ngraph;

builder::Tuple::Tuple(const Nodes& nodes)
    : Node("Tuple", Nodes{})
    , m_nodes(nodes)
{
}

std::shared_ptr<Node>
    builder::Tuple::copy_with_new_args(const std::vector<std::shared_ptr<Node>>& new_args) const
{
    return make_shared<Tuple>(new_args);
}

shared_ptr<Node> builder::Tuple::get_tuple_element(size_t i)
{
    return m_nodes.at(i);
}

shared_ptr<Node> builder::get_tuple_element(shared_ptr<Node> node, size_t i)
{
    shared_ptr<builder::Tuple> tuple = dynamic_pointer_cast<builder::Tuple>(node);
    if (tuple == nullptr)
    {
        throw ngraph_error("get_tuple_element called on a non-tuple");
    }
    return tuple->get_tuple_element(i);
}
