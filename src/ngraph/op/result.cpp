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
#include <typeindex>
#include <typeinfo>

#include "ngraph/node.hpp"
#include "ngraph/op/result.hpp"

using namespace std;
using namespace ngraph;

op::Result::Result(const shared_ptr<Node>& arg)
    : Op("Result", check_single_output_args({arg}))
{
    constructor_validate_and_infer_types();
}

void op::Result::validate_and_infer_types()
{
    NODE_VALIDATION_ASSERT(this, get_input_size() == 1) << "Argument has " << get_input_size()
                                                        << " outputs (1 expected).";

    // always borrow the placement conf even the default one
    set_placement_index(get_argument(0)->get_placement_index());
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::Result::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);

    auto res = make_shared<Result>(new_args.at(0));
    if (res)
    {
        res->set_needs_default_layout(m_needs_default_layout);
    }
    return res;
}

void op::Result::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    auto delta = deltas.at(0);

    adjoints.add_delta(get_argument(0), delta);
}
