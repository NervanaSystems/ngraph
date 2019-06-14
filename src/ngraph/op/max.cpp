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

#include "ngraph/op/max.hpp"
#include "ngraph/graph_util.hpp"

using namespace std;
using namespace ngraph;

const string op::Max::type_name{"Max"};

op::Max::Max()
{
}

op::Max::Max(const Output<Node>& arg, const AxisSet& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

op::Max::Max(const Output<Node>& arg, const Output<Node>& reduction_axes)
    : ArithmeticReduction(arg, reduction_axes)
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::Max::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Max>(new_args.at(0), new_args.at(1));
}

shared_ptr<Node> op::Max::get_default_value() const
{
    switch (get_element_type().get_type_enum())
    {
    case element::Type_t::boolean:
        return make_constant_from_string("0", get_element_type(), get_shape());
    case element::Type_t::bf16:
        return make_constant_from_string("-INFINITY", get_element_type(), get_shape());
    case element::Type_t::f32:
        return make_constant_from_string("-INFINITY", get_element_type(), get_shape());
    case element::Type_t::f64:
        return make_constant_from_string("-INFINITY", get_element_type(), get_shape());
    case element::Type_t::i8:
        return make_constant_from_string(
            to_string(numeric_limits<int8_t>::min()), get_element_type(), get_shape());
    case element::Type_t::i16:
        return make_constant_from_string(
            to_string(numeric_limits<int16_t>::min()), get_element_type(), get_shape());
    case element::Type_t::i32:
        return make_constant_from_string(
            to_string(numeric_limits<int32_t>::min()), get_element_type(), get_shape());
    case element::Type_t::i64:
        return make_constant_from_string(
            to_string(numeric_limits<int64_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u8:
        return make_constant_from_string(
            to_string(numeric_limits<uint8_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u16:
        return make_constant_from_string(
            to_string(numeric_limits<uint16_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u32:
        return make_constant_from_string(
            to_string(numeric_limits<uint32_t>::min()), get_element_type(), get_shape());
    case element::Type_t::u64:
        return make_constant_from_string(
            to_string(numeric_limits<uint64_t>::min()), get_element_type(), get_shape());
    default: throw runtime_error("Max default value not defined for type");
    }
}
