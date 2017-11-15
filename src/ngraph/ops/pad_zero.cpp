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

#include "ngraph/log.hpp"
#include "ngraph/ops/pad_zero.hpp"

using namespace std;
using namespace ngraph;

op::PadZero::PadZero(const Shape& padding_before,
                     const Shape& padding_after,
                     const std::shared_ptr<Node>& arg)
    : RequiresTensorViewArgs("PadZero", {arg})
    , m_padding_before(padding_before)
    , m_padding_after(padding_after)
{
    auto arg_tensor_view_type = m_inputs.at(0).get_tensor_view_type();
    auto& arg_element_type = arg_tensor_view_type->get_element_type();
    auto arg_shape = arg_tensor_view_type->get_shape();

    if (padding_before.size() != arg_shape.size())
    {
        throw ngraph_error("Padding shape rank must match the rank of the input");
    }

    if (padding_after.size() != arg_shape.size())
    {
        throw ngraph_error("Padding shape rank must match the rank of the input");
    }

    Shape out_shape;

    for (size_t i = 0; i < arg_shape.size(); i++)
    {
        out_shape.push_back(arg_shape[i] + padding_before[i] + padding_after[i]);
    }

    set_value_type_checked(make_shared<TensorViewType>(arg_element_type, out_shape));
}
