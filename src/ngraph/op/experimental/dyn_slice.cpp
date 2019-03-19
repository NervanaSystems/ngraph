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

#include "ngraph/op/experimental/dyn_slice.hpp"
#include <memory>

using namespace std;
using namespace ngraph;

op::DynSlice::DynSlice(const shared_ptr<Node>& arg,
                       const shared_ptr<Node>& lower_bounds,
                       const shared_ptr<Node>& upper_bounds,
                       const shared_ptr<Node>& strides)
    : Op("DynSlice", check_single_output_args({arg, lower_bounds, upper_bounds, strides}))
{
    constructor_validate_and_infer_types();
}

void op::DynSlice::validate_and_infer_types()
{
    auto lower_bounds_et = get_input_element_type(1);
    auto upper_bounds_et = get_input_element_type(2);
    auto strides_et = get_input_element_type(3);

    // check data types
    NODE_VALIDATION_CHECK(this,
                          lower_bounds_et.compatible(element::Type_t::i64),
                          "lower_bounds element type must be i64.");
    NODE_VALIDATION_CHECK(this,
                          upper_bounds_et.compatible(element::Type_t::i64),
                          "upper_bounds element type must be i64.");
    NODE_VALIDATION_CHECK(
        this, strides_et.compatible(element::Type_t::i64), "strides element type should be i64.");

    // check shapes
    auto arg_rank = get_input_partial_shape(0).rank();
    auto lower_bounds_rank = get_input_partial_shape(1).rank();
    auto upper_bounds_rank = get_input_partial_shape(2).rank();
    auto strides_rank = get_input_partial_shape(3).rank();
    NODE_VALIDATION_CHECK(this,
                          lower_bounds_rank.compatible(1),
                          "lower_bounds should have rank 1, got ",
                          lower_bounds_rank,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          upper_bounds_rank.compatible(1),
                          "upper_bounds should have rank 1, got ",
                          upper_bounds_rank,
                          ".");
    NODE_VALIDATION_CHECK(
        this, strides_rank.compatible(1), "strides should have rank 1, got ", strides_rank, ".");

    set_output_type(0, get_input_element_type(0), PartialShape::dynamic(arg_rank));
}

shared_ptr<Node> op::DynSlice::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<DynSlice>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
}

void op::DynSlice::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    throw ngraph_error("generate_adjoints not implemented for DynSlice");
}
