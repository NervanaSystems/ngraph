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

#include "ngraph/ops/constant.hpp"

using namespace ngraph::op;

void ConstantBase::propagate_types()
{
}

template <typename ET>
void check_value_strings(const std::vector<std::string>& value_strings)
{
    auto result = ET::read(value_strings);
}

void Constant::propagate_types()
{
    // No actual type propagation is done here; however, we check the number of value strings and
    // also call check_value_strings just to make sure the result will be parseable at compile
    // time. (It will throw an exception if not.)
    auto tvt = std::dynamic_pointer_cast<const TensorViewType>(get_value_type());
    if (nullptr == tvt)
    {
        throw ngraph_error("Constant does not have tensor view type");
    }
    auto shape = tvt->get_shape();

    if (ngraph::shape_size(shape) != m_value_strings.size())
    {
        throw ngraph_error("Constant does not have the expected number of literals");
    }

    auto& et = tvt->get_element_type();

    FUNCTION_ON_ELEMENT_TYPE(
        et, "Constant has unhandled element type", check_value_strings, m_value_strings);
}
