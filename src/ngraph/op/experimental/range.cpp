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

#include "ngraph/op/experimental/range.hpp"

using namespace std;
using namespace ngraph;

op::Range::Range(const shared_ptr<Node>& range_lo, const std::shared_ptr<Node>& range_hi)
    : Op("Range", check_single_output_args({range_lo,range_hi}))
{
    constructor_validate_and_infer_types();
}

void op::Range::validate_and_infer_types()
{
    element::Type et;
    NODE_VALIDATION_ASSERT(this, element::Type::merge(et, get_input_element_type(0), get_input_element_type(1)))
        << "Input element types are not compatible";
    NODE_VALIDATION_ASSERT(this, et.is_dynamic() || !et.is_real())
        << "Input element type must be an integral type";

    NODE_VALIDATION_ASSERT(this, get_input_partial_shape(0).rank().compatible(0))
        << "Input range_lo must be a scalar";
    NODE_VALIDATION_ASSERT(this, get_input_partial_shape(1).rank().compatible(0))
        << "Input range_hi must be a scalar";

    // Without static value propagation, all we know is that the output is a vector.
    set_output_type(0, get_input_element_type(0), PartialShape::dynamic(1));
}

shared_ptr<Node> op::Range::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::Range>(new_args.at(0), new_args.at(1));
}
