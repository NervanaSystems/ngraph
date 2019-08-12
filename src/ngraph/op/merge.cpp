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

#include "ngraph/op/merge.hpp"
#include "ngraph/log.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

const string op::Merge::type_name{"Merge"};

shared_ptr<Node> op::Merge::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Merge>(new_args.at(0), new_args.at(1), new_args.at(2));
}

op::Merge::Merge(const Output<Node>& cond, const Output<Node>& tval, const Output<Node>& fval)
    : Op({cond, tval, fval})
{
    constructor_validate_and_infer_types();
}

void op::Merge::validate_and_infer_types()
{
    const PartialShape& cond_shape = get_input_partial_shape(0);
    const PartialShape& tval_shape = get_input_partial_shape(1);
    const PartialShape& fval_shape = get_input_partial_shape(2);

    NODE_VALIDATION_CHECK(this,
                          cond_shape.rank().is_dynamic() || 0 == size_t(cond_shape.rank()),
                          "cond must be scalar");

    NODE_VALIDATION_CHECK(this,
                          element::boolean == get_input_element_type(0),
                          "cond must be of type element::boolean");
    element::Type result_et;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, get_input_element_type(1), get_input_element_type(2)),
        "Arguments do not have the same element type (tval element type: ",
        get_input_element_type(1),
        ", fval element type: ",
        get_input_element_type(2),
        ").");

    NODE_VALIDATION_CHECK(
        this, tval_shape.compatible(fval_shape), "tval and fval must have compatible shape");

    PartialShape output_shape{tval_shape};
    set_output_type(0, get_input_element_type(1), output_shape);
}
