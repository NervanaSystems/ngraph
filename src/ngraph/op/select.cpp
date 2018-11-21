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

#include <memory>

#include "ngraph/log.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/select.hpp"

using namespace std;
using namespace ngraph;

op::Select::Select(const shared_ptr<Node>& arg0,
                   const shared_ptr<Node>& arg1,
                   const shared_ptr<Node>& arg2)
    : Op("Select", check_single_output_args({arg0, arg1, arg2}))
{
    constructor_validate_and_infer_types();
}

void op::Select::validate_and_infer_types()
{
    NODE_VALIDATION_ASSERT(this,
                           get_input_element_type(0).is_dynamic() ||
                               get_input_element_type(0) == element::boolean)
        << "Argument 0 does not have boolean element type (element type: "
        << get_input_element_type(0) << ").";

    PartialShape result_shape = get_input_partial_shape(0);

    NODE_VALIDATION_ASSERT(this, PartialShape::merge_into(result_shape, get_input_partial_shape(1)))
        << "Argument shapes are inconsistent.";
    NODE_VALIDATION_ASSERT(this, PartialShape::merge_into(result_shape, get_input_partial_shape(2)))
        << "Argument shapes are inconsistent.";

    element::Type result_et;

    NODE_VALIDATION_ASSERT(
        this, element::Type::merge(result_et, get_input_element_type(1), get_input_element_type(2)))
        << "Argument 1 and 2 element types are inconsistent.";

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::Select::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Select>(new_args.at(0), new_args.at(1), new_args.at(2));
}

void op::Select::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto p = get_inputs().at(0).get_output().get_node();
    auto x = get_inputs().at(1).get_output().get_node();
    auto y = get_inputs().at(2).get_output().get_node();

    auto p_as_x_type = make_shared<op::Convert>(p, x->get_element_type());
    auto not_p_as_y_type = make_shared<op::Convert>(make_shared<op::Not>(p), y->get_element_type());

    adjoints.add_delta(x, delta * p_as_x_type);
    adjoints.add_delta(y, delta * not_p_as_y_type);
}
