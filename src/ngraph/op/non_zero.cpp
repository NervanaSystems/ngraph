//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/non_zero.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v2::NonZero::type_info;

op::v2::NonZero::NonZero(const Output<Node>& data)
    : Op({data})
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v2::NonZero::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v2::NonZero>(new_args.at(0));
}

void op::v2::NonZero::validate_and_infer_types()
{
    const auto data_ps = get_input_partial_shape(0);

    // NonZero produces tuples of size equal to input tensor rank
    PartialShape out_shape = {data_ps.rank(), Dimension::dynamic()};
    set_output_size(1);
    set_output_type(0, element::i64, out_shape);
}
