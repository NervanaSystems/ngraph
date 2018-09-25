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

#include <algorithm>
#include <iostream>

#include "ngraph/function.hpp"
#include "ngraph/op/dyn_reshape.hpp"
#include "ngraph/op/get_shape.hpp"

using namespace std;
using namespace ngraph;

op::DynReshape::DynReshape(const shared_ptr<Node>& arg, const shared_ptr<Node>& shape)
    : Op("DynReshape", check_single_output_args({arg, shape}))
{
    constructor_validate_and_infer_types();
}

void op::DynReshape::validate_and_infer_types()
{
    Shape input_shape = get_input_shape(0);

    NODE_VALIDATION_ASSERT(this, get_input_element_type(1) == element::i64)
        << "Shape argument must have type i64 (actual element type: " << get_input_element_type(1)
        << ").";

    // If the input shape tensor does not have a static value, we will set a fake shape.
    // Once incomplete shapes are supported through static value propagation, we will be able to
    // set "?" here.
    if (!get_inputs()[1].get_output().has_static_value())
    {
        set_output_type(0, get_input_element_type(0), Shape{});
        clear_output_static_value(0);
    }
    else
    {
        Shape output_shape = get_inputs()[1].get_output().get_static_value();

        // Once we have wildcard support we'll need to skip this check if input_shape or output_shape is not fully determined.
        NODE_VALIDATION_ASSERT(this, shape_size(input_shape) == shape_size(output_shape))
            << "Number of elements in output shape does not match number of elements in argument "
               "shape "
            << "(output shape: " << output_shape << ", argument shape: " << input_shape << ").";

        set_output_type(0, get_input_element_type(0), output_shape);

        // Static value propagation.
        // We will only propagate when reshaping to/from scalars/vectors. This has
        // the very useful property of being the identity. :/
        if (input_shape.size() < 2 && output_shape.size() < 2 &&
            get_inputs()[0].get_output().has_static_value())
        {
            auto& sv = get_inputs()[0].get_output().get_static_value();

            // This check should be redundant but you never know.
            NODE_VALIDATION_ASSERT(this, output_shape.size() == 1 || sv.size() == 1)
                << "Reshaping to a scalar but static value has more than one element "
                << "(input 0 static value: " << sv << ")";
            set_output_static_value(0, sv);
        }
        else
        {
            clear_output_static_value(0);
        }
    }
}

shared_ptr<Node> op::DynReshape::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynReshape>(new_args.at(0), new_args.at(1));
}

// TODO(amprocte): This is untested.
void op::DynReshape::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    auto x = get_argument(0);

    adjoints.add_delta(x, make_shared<op::DynReshape>(delta, make_shared<op::GetShape>(x)));
}
