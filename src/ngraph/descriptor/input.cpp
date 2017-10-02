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

#include "ngraph/ngraph.hpp"

using namespace std;
using namespace ngraph;
using namespace descriptor;

Input::Input(Node* node, size_t index, size_t argno, size_t arg_index, Output& output)
    : m_node(node)
    , m_index(index)
    , m_argno(argno)
    , m_arg_index(arg_index)
    , m_output(output)
{
    output.add_input(this);
}

std::shared_ptr<Node> Input::get_node()
{
    return m_node->shared_from_this();
}

const Tensor& Input::get_tensor() const
{
    return m_output.get_tensor();
}

Tensor& Input::get_tensor()
{
    return m_output.get_tensor();
}
