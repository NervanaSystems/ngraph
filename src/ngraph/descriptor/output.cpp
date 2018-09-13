//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/descriptor/output.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/node.hpp"

using namespace std;
using namespace ngraph;

descriptor::Output::Output(Node* node, size_t index, const shared_ptr<Tensor>& tensor)
    : m_node(node)
    , m_index(index)
    , m_tensor(tensor)
    , m_has_static_value(false)
    , m_static_value()
{
}

// Add an input to the vector of inputs that use this output.
void descriptor::Output::add_input(Input* input)
{
    m_inputs.insert(input);
}

void descriptor::Output::remove_input(Input* input)
{
    m_inputs.erase(input);
}

shared_ptr<Node> descriptor::Output::get_node() const
{
    return m_node->shared_from_this();
}

descriptor::Tensor& descriptor::Output::get_tensor() const
{
    return *m_tensor;
}

const Shape& descriptor::Output::get_shape() const
{
    return m_tensor->get_shape();
}

const element::Type& descriptor::Output::get_element_type() const
{
    return m_tensor->get_element_type();
}

const StaticValue& descriptor::Output::get_static_value() const
{
    if (!m_has_static_value)
    {
        throw ngraph_error("get_static_value() called on an output with no static value set");
    }
    return m_static_value;
}
