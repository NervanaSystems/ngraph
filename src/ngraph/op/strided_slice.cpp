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

#include "ngraph/op/strided_slice.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/validation_util.hpp"

#include <algorithm>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::StridedSlice::type_info;

op::v1::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const Output<Node>& strides,
                                   const std::vector<int64_t>& begin_mask,
                                   const std::vector<int64_t>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : Op({data, begin, end, strides})
    , m_begin_mask{begin_mask}
    , m_end_mask{end_mask}
    , m_new_axis_mask{new_axis_mask}
    , m_shrink_axis_mask{shrink_axis_mask}
    , m_ellipsis_mask{ellipsis_mask}
{
    constructor_validate_and_infer_types();
}

op::v1::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const std::vector<int64_t>& begin_mask,
                                   const std::vector<int64_t>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : StridedSlice(data,
                   begin,
                   end,
                   op::Constant::create(element::i64,
                                        Shape{begin_mask.size()},
                                        vector<int64_t>(begin_mask.size(), 1)),
                   begin_mask,
                   end_mask,
                   new_axis_mask,
                   shrink_axis_mask,
                   ellipsis_mask)
{
}

void op::v1::StridedSlice::validate_and_infer_types()
{
    const auto& begin_mask_et = get_input_element_type(1);
    const auto& end_mask_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          begin_mask_et.is_integral_number(),
                          "Begin mask must be an integral number, but is: ",
                          begin_mask_et);
    NODE_VALIDATION_CHECK(this,
                          end_mask_et.is_integral_number(),
                          "End mask must be an integral number, but is: ",
                          end_mask_et);

    auto are_mask_elem_in_range = [](size_t e) { return e == 0 || e == 1; };
    NODE_VALIDATION_CHECK(
        this,
        std::all_of(m_begin_mask.begin(), m_begin_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_end_mask.begin(), m_end_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_new_axis_mask.begin(), m_new_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(
                m_shrink_axis_mask.begin(), m_shrink_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_ellipsis_mask.begin(), m_ellipsis_mask.end(), are_mask_elem_in_range),
        "All masks of StridedSlice must have be 0 or 1");

    const vector<size_t> attr_sizes = {m_begin_mask.size(),
                                       m_end_mask.size(),
                                       m_new_axis_mask.size(),
                                       m_shrink_axis_mask.size(),
                                       m_ellipsis_mask.size()};
    const auto are_attr_sizes_eq =
        std::all_of(attr_sizes.begin(), attr_sizes.end(), [&attr_sizes](size_t s) {
            return (s == 0) || (attr_sizes[0] == s);
        });
    NODE_VALIDATION_CHECK(
        this, are_attr_sizes_eq, "All masks of StridedSlice must have the same size");

    const auto& data_rank = get_input_partial_shape(0).rank();
    const auto& begin_shape = get_input_partial_shape(1);
    if (begin_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              static_cast<size_t>(begin_shape.rank()) == 1,
                              "Begin input must be 1D (begin rank: ",
                              begin_shape.rank(),
                              ").");
    }
    const auto& end_shape = get_input_partial_shape(2);
    if (end_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              static_cast<size_t>(end_shape.rank()) == 1,
                              "End input must be 1D (end rank: ",
                              end_shape.rank(),
                              ").");
    }

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);

    auto begin_const = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr());
    auto end_const = as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    auto strides = as_type_ptr<op::Constant>(input_value(3).get_node_shared_ptr());

    if (begin_const && end_const && strides)
    {
        set_output_type(0,
                        get_input_element_type(0),
                        infer_slice_shape(this,
                                          get_input_partial_shape(0),
                                          begin_const->cast_vector<int64_t>(),
                                          end_const->cast_vector<int64_t>(),
                                          strides->cast_vector<int64_t>(),
                                          convert_mask_to_axis_set(get_begin_mask()),
                                          convert_mask_to_axis_set(get_end_mask()),
                                          convert_mask_to_axis_set(get_new_axis_mask()),
                                          convert_mask_to_axis_set(get_shrink_axis_mask()),
                                          convert_mask_to_axis_set(get_ellipsis_mask())));
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(data_rank));
    }
}

AxisSet op::v1::StridedSlice::convert_mask_to_axis_set(const std::vector<int64_t>& mask) const
{
    AxisSet axis_set{};
    for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i)
    {
        if (mask[i] == 1)
        {
            axis_set.emplace(i);
        }
    }
    return axis_set;
}

shared_ptr<Node> op::v1::StridedSlice::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::StridedSlice>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         m_begin_mask,
                                         m_end_mask,
                                         m_new_axis_mask,
                                         m_shrink_axis_mask,
                                         m_ellipsis_mask);
}

void op::v1::StridedSlice::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                             const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for StridedSlice");
}

constexpr NodeTypeInfo op::v2::StridedSlice::type_info;
op::v2::StridedSlice::StridedSlice(const Output<Node>& data,
                                   const Output<Node>& begin,
                                   const Output<Node>& end,
                                   const Output<Node>& axes,
                                   const Output<Node>& strides,
                                   const Output<Node>& begin_mask,
                                   const Output<Node>& end_mask,
                                   const std::vector<int64_t>& new_axis_mask,
                                   const std::vector<int64_t>& shrink_axis_mask,
                                   const std::vector<int64_t>& ellipsis_mask)
    : Op({data, begin, end, axes, strides, begin_mask, end_mask})
    , m_new_axis_mask{new_axis_mask}
    , m_shrink_axis_mask{shrink_axis_mask}
    , m_ellipsis_mask{ellipsis_mask}
{
    constructor_validate_and_infer_types();
}

void op::v2::StridedSlice::validate_and_infer_types()
{
    auto are_mask_elem_in_range = [this](size_t e) { return e == 0 || e == 1; };

    auto get_valid_array_idx = [&](int64_t idx, int64_t last_idx) {
        return (idx >= 0) ? std::min(idx, last_idx) : std::max<int64_t>(0, last_idx + idx);
    };

    auto negative_axis_converter = [&](const std::vector<int64_t>& data_vec,
                                       const std::vector<int64_t>& axis_vec,
                                       ngraph::Shape shape) {
        std::vector<int64_t> bounds = data_vec;
        for (size_t idx = 0; idx < axis_vec.size(); ++idx)
        {
            size_t axis = axis_vec.at(idx);
            bounds.at(axis) = get_valid_array_idx(data_vec.at(idx), shape.at(axis));
        }
        return bounds;
    };

    const auto& begin_mask_et = get_input_element_type(1);
    const auto& end_mask_et = get_input_element_type(2);

    const auto& begin_mask_et_ptr = input_value(5).get_node_shared_ptr();
    const auto& end_mask_et_ptr = input_value(6).get_node_shared_ptr();
    auto m_begin_mask_et = as_type_ptr<op::Constant>(begin_mask_et_ptr)->cast_vector<int64_t>();
    auto m_end_mask_et = as_type_ptr<op::Constant>(end_mask_et_ptr)->cast_vector<int64_t>();

    NODE_VALIDATION_CHECK(this,
                          begin_mask_et.is_integral_number(),
                          "Begin mask must be an integral number, but is: ",
                          begin_mask_et);
    NODE_VALIDATION_CHECK(this,
                          end_mask_et.is_integral_number(),
                          "End mask must be an integral number, but is: ",
                          end_mask_et);

    NODE_VALIDATION_CHECK(
        this,
        std::all_of(m_begin_mask_et.begin(), m_begin_mask_et.end(), are_mask_elem_in_range) &&
            std::all_of(m_end_mask_et.begin(), m_end_mask_et.end(), are_mask_elem_in_range) &&
            std::all_of(m_new_axis_mask.begin(), m_new_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(
                m_shrink_axis_mask.begin(), m_shrink_axis_mask.end(), are_mask_elem_in_range) &&
            std::all_of(m_ellipsis_mask.begin(), m_ellipsis_mask.end(), are_mask_elem_in_range),
        "All masks of StridedSlice must have be 0 or 1");

    const vector<size_t> attr_sizes = {m_begin_mask_et.size(),
                                       m_end_mask_et.size(),
                                       m_new_axis_mask.size(),
                                       m_shrink_axis_mask.size(),
                                       m_ellipsis_mask.size()};
    const auto are_attr_sizes_eq =
        std::all_of(attr_sizes.begin(), attr_sizes.end(), [&attr_sizes](size_t s) {
            return (s == 0) || (attr_sizes[0] == s);
        });
    NODE_VALIDATION_CHECK(
        this, are_attr_sizes_eq, "All masks of StridedSlice must have the same size");

    const auto& data_rank = get_input_partial_shape(0).rank();
    const auto& begin_shape = get_input_partial_shape(1);

    if (begin_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              static_cast<size_t>(begin_shape.rank()) == 1,
                              "Begin input must be 1D (begin rank: ",
                              begin_shape.rank(),
                              ").");
    }
    const auto& end_shape = get_input_partial_shape(2);
    if (end_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(this,
                              static_cast<size_t>(end_shape.rank()) == 1,
                              "End input must be 1D (end rank: ",
                              end_shape.rank(),
                              ").");
    }

    set_input_is_relevant_to_shape(1);
    set_input_is_relevant_to_shape(2);
    set_input_is_relevant_to_shape(3);
    set_input_is_relevant_to_shape(4);

    auto begin_const = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr());
    auto end_const = as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());
    auto axes = as_type_ptr<op::Constant>(input_value(3).get_node_shared_ptr());
    auto strides = as_type_ptr<op::Constant>(input_value(4).get_node_shared_ptr());

    if (begin_const && end_const && strides && axes)
    {
        auto shapes = input_value(0);

        auto lower_bounds = negative_axis_converter(
            begin_const->cast_vector<int64_t>(), axes->cast_vector<int64_t>(), shapes.get_shape());
        auto upper_bounds = negative_axis_converter(
            end_const->cast_vector<int64_t>(), axes->cast_vector<int64_t>(), shapes.get_shape());

        for (size_t idx = 0; idx < lower_bounds.size(); ++idx)
        {
            if (lower_bounds.at(idx) > upper_bounds.at(idx))
            {
                upper_bounds.at(idx) = lower_bounds.at(idx);
            }
        }

        set_output_type(0,
                        get_input_element_type(0),
                        infer_slice_shape(this,
                                          get_input_partial_shape(0),
                                          lower_bounds,
                                          upper_bounds,
                                          strides->cast_vector<int64_t>(),
                                          convert_mask_to_axis_set(m_begin_mask_et),
                                          convert_mask_to_axis_set(m_end_mask_et),
                                          convert_mask_to_axis_set(get_new_axis_mask()),
                                          convert_mask_to_axis_set(get_shrink_axis_mask()),
                                          convert_mask_to_axis_set(get_ellipsis_mask())));
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(data_rank));
    }
}

AxisSet op::v2::StridedSlice::convert_mask_to_axis_set(const std::vector<int64_t>& mask) const
{
    AxisSet axis_set{};
    for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i)
    {
        if (mask[i] == 1)
        {
            axis_set.emplace(i);
        }
    }
    return axis_set;
}

shared_ptr<Node> op::v2::StridedSlice::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v2::StridedSlice>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         new_args.at(5),
                                         new_args.at(6),
                                         m_new_axis_mask,
                                         m_shrink_axis_mask,
                                         m_ellipsis_mask);
}

void op::v2::StridedSlice::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                             const OutputVector& /* deltas */)
{
    throw ngraph_error("generate_adjoints not implemented for StridedSlice");
}
