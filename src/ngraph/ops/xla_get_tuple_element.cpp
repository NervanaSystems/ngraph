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
#include <sstream>

#include "ngraph/ops/xla_get_tuple_element.hpp"
#include "ngraph/ops/xla_tuple.hpp"

using namespace std;
using namespace ngraph;

op::XLAGetTupleElement::XLAGetTupleElement(const std::shared_ptr<Node>& arg, size_t n)
    : XLANode("XLAGetTupleElement", {arg})
    , m_n{n}
{
    m_arg = dynamic_pointer_cast<XLANode>(arg);
    if (m_arg == nullptr || m_arg->get_tuple_value() == nullptr)
    {
        throw ngraph_error("Argument must be a tuple view");
    }

    const Nodes& elements = m_arg->get_tuple_elements();

    if (m_n >= elements.size())
    {
        throw ngraph_error("Indexing tuple beyond its size");
    }
}

Nodes op::XLAGetTupleElement::get_input_ops() //const
{
    return Nodes{m_arg};
}

shared_ptr<const op::XLATuple> op::XLAGetTupleElement::get_tuple_value() const
{
    return dynamic_pointer_cast<const op::XLATuple>(m_arg->get_tuple_elements().at(m_n));
}

const Nodes& op::XLAGetTupleElement::get_tuple_elements() const
{
    return get_tuple_value()->get_tuple_elements();
}
