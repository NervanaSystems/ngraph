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

#include "ngraph/ops/sum.hpp"
#include "ngraph/function.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph::op;

void Sum::propagate_types()
{
    if (m_arguments.size() != 1)
    {
        throw ngraph_error("Wrong number of arguments.");
    }

    auto st = get_shape_et(m_arguments.at(0));
    if (st.type == element::Bool::element_type())
    {
        throw ngraph_error("Argument for sum must have numeric element type");
    }

    for (auto axis : m_reduction_axes)
    {
        if (axis >= st.shape.size())
        {
            throw ngraph_error("Reduction axis for sum is out of bounds");
        }
    }

    Shape result_shape;

    for (size_t i = 0; i < st.shape.size(); i++)
    {
        if (m_reduction_axes.count(i) == 0)
        {
            result_shape.push_back(st.shape.at(i));
        }
    }

    set_value_type_checked(make_shared<TensorViewType>(st.type, result_shape));
}
