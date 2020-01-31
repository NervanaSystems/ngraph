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

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::TopK::type_info;

op::v0::TopK::TopK(const Output<Node>& arg,
                   size_t top_k_axis,
                   const element::Type& index_element_type,
                   size_t k,
                   bool compute_max,
                   SortType sort)
    : Op({arg})
    , m_index_element_type(index_element_type)
    , m_compute_max(compute_max)
    , m_sort(sort)
{
    set_argument(1, op::Constant::create(element::i64, Shape{1}, {k})->output(0));
    set_argument(2, op::Constant::create(element::i64, Shape{1}, {top_k_axis})->output(0));
    add_provenance_group_member(input_value(1).get_node_shared_ptr());
    add_provenance_group_member(input_value(2).get_node_shared_ptr());
    constructor_validate_and_infer_types();
}

op::v0::TopK::TopK(const Output<Node>& arg,
                   const Output<Node>& k,
                   size_t top_k_axis,
                   const element::Type& index_element_type,
                   bool compute_max,
                   SortType sort)
    : Op({arg, k})
    , m_index_element_type(index_element_type)
    , m_compute_max(compute_max)
    , m_sort(sort)
{
    set_argument(2, op::Constant::create(element::i64, Shape{1}, {top_k_axis})->output(0));
    add_provenance_group_member(input_value(2).get_node_shared_ptr());
    constructor_validate_and_infer_types();
}

op::v0::TopK::TopK(const Output<Node>& arg,
                   const Output<Node>& k,
                   const Output<Node>& top_k_axis,
                   const element::Type& index_element_type,
                   bool compute_max,
                   SortType sort)
    : Op({arg, k, top_k_axis})
    , m_index_element_type(index_element_type)
    , m_compute_max(compute_max)
    , m_sort(sort)
{
    constructor_validate_and_infer_types();
}

size_t op::v0::TopK::get_k() const
{
    size_t k = 0;
    if (auto const_op = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr()))
    {
        k = const_op->get_vector<int64_t>()[0];
    }
    Dimension top_k_axis = get_top_k_axis_dynamic();
    if (k == 0 && get_input_partial_shape(0).is_static() && top_k_axis.is_static())
    {
        k = get_input_partial_shape(0).to_shape()[static_cast<size_t>(top_k_axis)];
    }
    return k;
}

void op::v0::TopK::set_k(size_t k)
{
    shared_ptr<Node> current_const =
        get_input_size() == 1 ? nullptr : input_value(1).get_node_shared_ptr();
    auto replacement_const = op::Constant::create(element::i64, Shape{1}, {k})->output(0);
    this->input(1).replace_source_output(replacement_const);
    replace_provenance_group_member(current_const, replacement_const.get_node_shared_ptr());
}

size_t op::v0::TopK::get_top_k_axis() const
{
    auto d = get_top_k_axis_dynamic();
    NGRAPH_CHECK(d.is_static(),
                 "get_top_k_axis called on a TopK node whose 'top_k_axis' input is not constant");
    return static_cast<size_t>(d);
}

Dimension op::v0::TopK::get_top_k_axis_dynamic() const
{
    auto const_op = dynamic_pointer_cast<op::Constant>(input_value(2).get_node_shared_ptr());
    if (const_op)
    {
        return const_op->get_vector<int64_t>()[0];
    }
    else
    {
        return Dimension::dynamic();
    }
}

void op::v0::TopK::set_top_k_axis(size_t top_k_axis)
{
    shared_ptr<Node> current_const = input_value(2).get_node_shared_ptr();
    auto replacement_const = op::Constant::create(element::i64, Shape{1}, {top_k_axis})->output(0);
    this->input(2).replace_source_output(replacement_const);
    replace_provenance_group_member(current_const, replacement_const.get_node_shared_ptr());
}

void op::v0::TopK::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);
    Rank input_rank = input_shape.rank();
    element::Type input_element_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(
        this, !m_index_element_type.is_dynamic(), "Argument element type must not be dynamic.");

    NODE_VALIDATION_CHECK(this,
                          m_index_element_type == element::i32 ||
                              m_index_element_type == element::i64,
                          "Argument element type must be i64 or i32 (got ",
                          m_index_element_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || static_cast<size_t>(input_rank) > 0,
                          "Argument rank must be greater than 0.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).compatible(element::i64),
                          "Element type for 'k' must be i64");
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).compatible(element::i64),
                          "Element type for 'top_k_axis' must be i64");

    Dimension top_k_axis = get_top_k_axis_dynamic();
    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || top_k_axis.is_dynamic() ||
                              static_cast<size_t>(top_k_axis) < static_cast<size_t>(input_rank),
                          "TopK axis (",
                          top_k_axis,
                          ") is out of bounds.");

    size_t k = get_k();
    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || top_k_axis.is_dynamic() ||
                              input_shape[static_cast<size_t>(top_k_axis)].is_dynamic() ||
                              static_cast<size_t>(k) <=
                                  static_cast<size_t>(input_shape[static_cast<size_t>(top_k_axis)]),
                          "K (",
                          k,
                          ") exceeds the dimension (",
                          input_shape[static_cast<size_t>(top_k_axis)],
                          ") of the TopK axis (axis ",
                          top_k_axis,
                          ").");

    PartialShape output_shape{input_shape};

    if (input_rank.is_static())
    {
        if (top_k_axis.is_static())
        {
            if (k != 0)
            {
                output_shape[static_cast<size_t>(top_k_axis)] = k;
            }
            else if (k == 0 && output_shape[static_cast<size_t>(top_k_axis)].is_static())
            {
                output_shape[static_cast<size_t>(top_k_axis)] =
                    input_shape[static_cast<size_t>(top_k_axis)];
            }
        }
        else
        {
            // If top_k_axis is not static and k is not 0, then we could be changing any
            // dimension. So we have to change all dimensions to dynamic.
            output_shape = PartialShape::dynamic(input_rank);
        }
    }

    set_input_is_relevant_to_shape(2);

    set_output_size(2);
    set_output_type(0, m_index_element_type, output_shape);
    set_output_type(1, input_element_type, output_shape);
}

shared_ptr<Node> op::v0::TopK::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<TopK>(new_args.at(0),
                             new_args.at(1),
                             new_args.at(2),
                             m_index_element_type,
                             m_compute_max,
                             m_sort);
}

void op::v0::TopK::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                     const OutputVector& /* deltas */)
{
    throw ngraph_error("Forward-propagation-only operation");
}

// v1 version starts
constexpr NodeTypeInfo op::v1::TopK::type_info;

op::v1::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const std::string& mode,
                   const std::string& sort,
                   const element::Type& index_element_type)
    : Op{{data, k}}
    , m_axis{axis}
    , m_mode{mode_from_string(mode)}
    , m_sort{sort_type_from_string(sort)}
    , m_index_element_type{index_element_type}
{
    constructor_validate_and_infer_types();
}

op::v1::TopK::TopK(const Output<Node>& data,
                   const Output<Node>& k,
                   const int64_t axis,
                   const Mode mode,
                   const SortType sort,
                   const element::Type& index_element_type)
    : Op{{data, k}}
    , m_axis{axis}
    , m_mode{mode}
    , m_sort{sort}
    , m_index_element_type{index_element_type}
{
    constructor_validate_and_infer_types();
}

void op::v1::TopK::validate_and_infer_types()
{
    const auto& input_partial_shape = get_input_partial_shape(0);
    const auto input_rank = input_partial_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          input_rank.is_dynamic() || static_cast<size_t>(input_rank) > 0,
                          "Input rank must be greater than 0.");

    const auto& k_partial_shape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(
        this, k_partial_shape.rank().compatible(0), "The 'K' input must be a scalar.");

    size_t k = 0;
    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        k = read_k_from_constant_node(input_value(1).get_node_shared_ptr(),
                                      get_input_element_type(1));
    }

    PartialShape output_shape{input_partial_shape};

    if (output_shape.rank().is_static())
    {
        NODE_VALIDATION_CHECK(
            this,
            m_axis >= 0 && static_cast<size_t>(m_axis) < static_cast<size_t>(output_shape.rank()),
            "TopK axis (",
            m_axis,
            ") is out of bounds.");

        if (k != 0)
        {
            output_shape[m_axis] = k;
        }
    }

    set_output_size(2);
    set_output_type(0, get_input_element_type(0), output_shape);
    set_output_type(1, m_index_element_type, output_shape);
}

size_t op::v1::TopK::read_k_from_constant_node(const shared_ptr<Node>& node,
                                               const element::Type& k_element_type) const
{
    NODE_VALIDATION_CHECK(this,
                          k_element_type == element::i8 || k_element_type == element::i32 ||
                              k_element_type == element::i64,
                          "K input element type must be i8, i32 or i64 (got ",
                          k_element_type,
                          ").");

    const auto k_constant = as_type_ptr<op::Constant>(node);

    size_t k = 0;

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (static_cast<element::Type_t>(k_element_type))
    {
    case element::Type_t::i8: k = validate_and_get_k<int8_t>(k_constant); break;
    case element::Type_t::i32: k = validate_and_get_k<int32_t>(k_constant); break;
    case element::Type_t::i64: k = validate_and_get_k<int64_t>(k_constant); break;
    default: break;
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    return k;
}

template <typename T>
size_t op::v1::TopK::validate_and_get_k(const shared_ptr<op::Constant>& k_constant) const
{
    const auto k_const_contents = k_constant->get_vector<T>();

    NODE_VALIDATION_CHECK(this,
                          k_const_contents.size() == 1,
                          "Only one value (scalar) should be provided as the 'K' input to TopK",
                          " (got ",
                          k_const_contents.size(),
                          " elements).");

    NODE_VALIDATION_CHECK(this,
                          k_const_contents[0] > 0,
                          "The value of 'K' must be a positive number.",
                          " (got ",
                          k_const_contents[0],
                          ").");

    return static_cast<size_t>(k_const_contents[0]);
}

void op::v1::TopK::generate_adjoints(autodiff::Adjoints& /*adjoints*/,
                                     const OutputVector& /* deltas */)
{
    throw ngraph_error("Forward-propagation-only operation");
}

shared_ptr<Node> op::v1::TopK::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    auto new_v1_topk =
        make_shared<v1::TopK>(new_args.at(0), new_args.at(1), m_axis, m_mode, m_sort);

    new_v1_topk->set_index_element_type(m_index_element_type);

    return std::move(new_v1_topk);
}

op::v1::TopK::Mode op::v1::TopK::mode_from_string(const std::string& mode) const
{
    static const std::map<std::string, Mode> allowed_values = {{"max", Mode::MAX},
                                                               {"min", Mode::MIN}};

    NODE_VALIDATION_CHECK(this, allowed_values.count(mode) > 0, "Invalid 'mode' value passed in.");

    return allowed_values.at(mode);
}

op::v1::TopK::SortType op::v1::TopK::sort_type_from_string(const std::string& sort) const
{
    static const std::map<std::string, SortType> allowed_values = {
        {"none", SortType::NONE},
        {"index", SortType::SORT_INDICES},
        {"value", SortType::SORT_VALUES}};

    NODE_VALIDATION_CHECK(this, allowed_values.count(sort) > 0, "Invalid 'sort' value passed in.");

    return allowed_values.at(sort);
}

size_t op::v1::TopK::get_k() const
{
    size_t k = 0;
    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        k = read_k_from_constant_node(input_value(1).get_node_shared_ptr(),
                                      get_input_element_type(1));
    }

    if (k == 0 && get_input_partial_shape(0).is_static())
    {
        k = get_input_partial_shape(0).to_shape()[m_axis];
    }
    return k;
}

void op::v1::TopK::set_k(size_t k)
{
    this->input(1).replace_source_output(
        op::Constant::create(element::i64, Shape{}, {k})->output(0));
}
