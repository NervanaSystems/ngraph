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

#include <algorithm>

#include "ngraph/op/constant.hpp"
#include "ngraph/op/experimental/range.hpp"

using namespace std;
using namespace ngraph;

const string op::Range::type_name = "Range";

op::Range::Range(const Output<Node>& start, const Output<Node>& stop, const Output<Node>& step)
    : Op({start, stop, step})
{
    constructor_validate_and_infer_types();
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, void>::type
    check_start(const op::Range* node, T start)
{
    // Nothing to check for integral types.
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, void>::type
    check_stop(const op::Range* node, T stop)
{
    // Nothing to check for integral types.
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, void>::type
    check_step(const op::Range* node, T step)
{
    NODE_VALIDATION_CHECK(node, step != 0, "'step' cannot be zero.");
}

//
// The code in the following three functions is a bit awkward, to work around some compiler
// warnings and the need to support our custom float16/bfloat16 type:
//
// (1) We can't use STL things like isnan, because our custom float16/bfloat16 types don't always
//     support them.
// (2) We check whether (x - x) == (x - x) to check for "is_finite".
// (3) We have to break (x - x) out into a temporary because otherwise the compiler throws a
//     warning about == on floats.
// (4) We check <0 || >0 to check for != 0, because otherwise the compiler throws a warning about
//     == on floats.
//
template <typename T>
static
    typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                std::is_same<T, bfloat16>::value,
                            void>::type
    check_start(const op::Range* node, T start)
{
    T start_minus_start = start - start;
    NODE_VALIDATION_CHECK(node,
                          start == start && start_minus_start == start_minus_start,
                          "'start' cannot be nan or infinite.");
}

template <typename T>
static
    typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                std::is_same<T, bfloat16>::value,
                            void>::type
    check_stop(const op::Range* node, T stop)
{
    T stop_minus_stop = stop - stop;
    NODE_VALIDATION_CHECK(node,
                          stop == stop && stop_minus_stop == stop_minus_stop,
                          "'stop' cannot be nan or infinite.");
}

template <typename T>
static
    typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                std::is_same<T, bfloat16>::value,
                            void>::type
    check_step(const op::Range* node, T step)
{
    T step_minus_step = step - step;
    NODE_VALIDATION_CHECK(node,
                          step == step && step_minus_step == step_minus_step &&
                              (step > static_cast<T>(0) || step < static_cast<T>(0)),
                          "'step' cannot be zero, nan, or infinite.");
}

template <typename T>
static typename std::enable_if<std::is_integral<T>::value, T>::type adjust_for_step_and_sign(T span,
                                                                                             T step)
{
    return ceil_div(span < 0 ? -span : span, step < 0 ? -step : step);
}

template <typename T>
static
    typename std::enable_if<std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                std::is_same<T, bfloat16>::value,
                            T>::type
    adjust_for_step_and_sign(T span, T step)
{
    return ceil(fabs(span) / fabs(step));
}

template <typename T>
static PartialShape infer_output_shape(const op::Range* node, const element::Type& et)
{
    auto const_start =
        dynamic_pointer_cast<op::Constant>(node->input_value(0).get_node_shared_ptr());
    auto const_stop =
        dynamic_pointer_cast<op::Constant>(node->input_value(1).get_node_shared_ptr());
    auto const_step =
        dynamic_pointer_cast<op::Constant>(node->input_value(2).get_node_shared_ptr());

    T start = static_cast<T>(0);
    T stop = static_cast<T>(0);
    T step = static_cast<T>(0);

    if (const_start != nullptr)
    {
        std::vector<T> start_val = const_start->get_vector<T>();
        NODE_VALIDATION_CHECK(node, start_val.size() == 1);
        start = start_val[0];
        check_start<T>(node, start);
    }

    if (const_stop != nullptr)
    {
        std::vector<T> stop_val = const_stop->get_vector<T>();
        NODE_VALIDATION_CHECK(node, stop_val.size() == 1);
        stop = stop_val[0];
        check_stop<T>(node, stop);
    }

    if (const_step != nullptr)
    {
        std::vector<T> step_val = const_step->get_vector<T>();
        NODE_VALIDATION_CHECK(node, step_val.size() == 1);
        step = step_val[0];
        check_step<T>(node, step);
    }

    PartialShape result{PartialShape::dynamic(1)};

    if (const_start != nullptr && const_stop != nullptr && const_step != nullptr)
    {
        T span;

        if (step > static_cast<T>(0) && start >= stop)
        {
            span = static_cast<T>(0);
        }
        else if (step < static_cast<T>(0) && start <= stop)
        {
            span = static_cast<T>(0);
        }
        else
        {
            span = stop - start;
        }

        T strided = adjust_for_step_and_sign<T>(span, step);

        result = PartialShape{Dimension(static_cast<int64_t>(strided))};
    }

    return result;
}

void op::Range::validate_and_infer_types()
{
    set_input_is_relevant_to_shape(0);
    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);

    auto result_et = element::dynamic;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, result_et, get_input_element_type(0)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(1)) &&
            element::Type::merge(result_et, result_et, get_input_element_type(2)),
        "Element types for start, stop, and step do not match.");

    NODE_VALIDATION_CHECK(this,
                          result_et != element::boolean,
                          "Element type for start, stop, and step, must not be boolean.");

    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(0).compatible(Shape{}), "'start' input is not a scalar");
    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(0).compatible(Shape{}), "'stop' input is not a scalar");
    NODE_VALIDATION_CHECK(
        this, get_input_partial_shape(0).compatible(Shape{}), "'step' input is not a scalar");

    PartialShape result_shape;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
    switch (result_et)
    {
    case element::Type_t::bf16: result_shape = infer_output_shape<bfloat16>(this, result_et); break;
    case element::Type_t::f16: result_shape = infer_output_shape<float16>(this, result_et); break;
    case element::Type_t::f32: result_shape = infer_output_shape<float>(this, result_et); break;
    case element::Type_t::f64: result_shape = infer_output_shape<double>(this, result_et); break;
    case element::Type_t::i8: result_shape = infer_output_shape<int8_t>(this, result_et); break;
    case element::Type_t::i16: result_shape = infer_output_shape<int16_t>(this, result_et); break;
    case element::Type_t::i32: result_shape = infer_output_shape<int32_t>(this, result_et); break;
    case element::Type_t::i64: result_shape = infer_output_shape<int64_t>(this, result_et); break;
    case element::Type_t::u8: result_shape = infer_output_shape<uint8_t>(this, result_et); break;
    case element::Type_t::u16: result_shape = infer_output_shape<uint16_t>(this, result_et); break;
    case element::Type_t::u32: result_shape = infer_output_shape<uint32_t>(this, result_et); break;
    case element::Type_t::u64: result_shape = infer_output_shape<uint64_t>(this, result_et); break;
    case element::Type_t::dynamic: result_shape = PartialShape::dynamic(1); break;
    case element::Type_t::undefined:
    case element::Type_t::boolean:
        NODE_VALIDATION_CHECK(
            this, false, "Internal nGraph error: unsupported element type: ", result_et);
        break;
    }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

    set_output_type(0, result_et, result_shape);
}

shared_ptr<Node> op::Range::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Range>(new_args.at(0), new_args.at(1), new_args.at(2));
}
