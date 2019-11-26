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

#include <memory>

#include "ngraph/op/convert_like.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::ConvertLike::type_info;

op::v1::ConvertLike::ConvertLike(const Output<Node>& data, const Output<Node>& like)
    : Op({data, like})
{
    constructor_validate_and_infer_types();
}

void op::v1::ConvertLike::validate_and_infer_types()
{
    set_output_type(0, get_input_element_type(1), get_input_partial_shape(0));
}

shared_ptr<Node> op::v1::ConvertLike::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<ConvertLike>(new_args.at(0), new_args.at(1));
}

void op::v1::ConvertLike::generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas)
{
    const auto delta = deltas.at(0);

    adjoints.add_delta(input_value(0), make_shared<op::v1::ConvertLike>(delta, input_value(1)));
}
