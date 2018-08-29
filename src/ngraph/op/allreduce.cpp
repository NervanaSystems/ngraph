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

#include "ngraph/op/allreduce.hpp"

using namespace std;
using namespace ngraph;

op::AllReduce::AllReduce(const shared_ptr<Node>& arg)
    : Op("AllReduce", check_single_output_args({arg}))
{
    constructor_validate_and_infer_types();
}

void op::AllReduce::validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(0), get_input_shape(0));

    if ((get_input_element_type(0) != element::f32) && (get_input_element_type(0) != element::f64))
    {
        throw ngraph_error("Unsupported data type for AllReduce");
    }
}

shared_ptr<Node> op::AllReduce::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 1)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<AllReduce>(new_args.at(0));
}
