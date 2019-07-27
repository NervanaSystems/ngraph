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

#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

const string op::ShapeOf::type_name{"ShapeOf"};

op::ShapeOf::ShapeOf(const shared_ptr<Node>& arg)
    : Op(check_single_output_args({arg}))
{
    constructor_validate_and_infer_types();
}

void op::ShapeOf::validate_and_infer_types()
{
    set_input_is_relevant_to_value(0, false);
    set_output_type(0, element::i64, PartialShape{get_input_partial_shape(0).rank()});
}

shared_ptr<Node> op::ShapeOf::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ShapeOf>(new_args.at(0));
}
