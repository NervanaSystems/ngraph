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

#include <sstream>

#include "ngraph/ops/get_tuple_element.hpp"

using namespace std;
using namespace ngraph::op;

void GetTupleElement::propagate_types()
{
    if (m_arguments.size() != 1)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto arg0_tuple_type =
        dynamic_pointer_cast<const TupleType>(m_arguments.at(0)->get_value_type());
    if (nullptr == arg0_tuple_type)
    {
        throw ngraph_error("Argument must be a tuple view");
    }

    if (m_n >= arg0_tuple_type->get_element_types().size())
    {
        throw ngraph_error("Indexing tuple beyond its size");
    }

    set_value_type_checked(arg0_tuple_type->get_element_types().at(m_n));
}
