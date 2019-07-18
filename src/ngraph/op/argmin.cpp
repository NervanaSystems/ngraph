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

#include "ngraph/op/argmin.hpp"

using namespace std;
using namespace ngraph;

const string op::ArgMin::type_name{"ArgMin"};

op::ArgMin::ArgMin(const Output<Node>& arg, size_t axis, const element::Type& index_element_type)
    : op::util::IndexReduction(arg, axis, index_element_type)
{
    constructor_validate_and_infer_types();
}

void op::ArgMin::validate_reduction() const
{
    PartialShape input_shape = get_input_partial_shape(0);
    Rank rank = input_shape.rank();
    if (!rank.is_dynamic())
    {
        Dimension d = input_shape[m_axis];
        if (d.is_static())
        {
            NODE_VALIDATION_CHECK(this,
                                  0 != size_t(d),
                                  "Tensor reduction axis can not be empty, shape is: ",
                                  input_shape);
        }
    }
}

shared_ptr<Node> op::ArgMin::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ArgMin>(new_args.at(0), m_axis, this->get_element_type());
}

std::shared_ptr<Node> op::ArgMin::get_default_value() const
{
    // Choice of value here is arbitrary, because validation should be rejecting cases where the
    // axis of reduction has size zero.
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
