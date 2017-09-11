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

Output::Output(Node* node, size_t index, const std::shared_ptr<TensorView>& tensor_view)
    : m_node(node)
    , m_index(index)
    , m_tensor_view(tensor_view)
{
}

// Add an input to the vector of inputs that use this output.
void Output::add_input(Input* input)
{
    m_inputs.insert(input);
}

std::shared_ptr<Node> Output::get_node() const
{
    return m_node->shared_from_this();
}
