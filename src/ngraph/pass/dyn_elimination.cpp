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

#include <numeric>

#include "dyn_elimination.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_replace_slice.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/range.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/transpose.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"
#include "ngraph/runtime/reference/range.hpp"
#include "ngraph/slice_plan.hpp"

using namespace std;
using namespace ngraph;

pass::DynElimination::DynElimination()
    : GraphRewrite()
{
    construct_transpose();
    construct_dyn_broadcast();
    construct_dyn_replace_slice();
    construct_dyn_slice();
    construct_range();
}

void pass::DynElimination::construct_transpose()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto perm_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());

    auto transpose = make_shared<op::v1::Transpose>(data_arg_label, perm_arg_label);

    auto transpose_callback = [data_arg_label, perm_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto data_arg = pattern_map[data_arg_label];
        auto perm_arg = static_pointer_cast<op::v0::Constant>(pattern_map[perm_arg_label]);

        // TODO(amprocte): Can't handle the case where data shape is dynamic, because static
        // Reshape requries the exact output shape to be declared. See if we can come up with a
        // workaround.
        if (data_arg->get_output_partial_shape(0).is_dynamic())
        {
            return false;
        }

        auto& data_shape = data_arg->get_output_shape(0);

        NGRAPH_CHECK(perm_arg->get_output_partial_shape(0).rank().compatible(1));
        NGRAPH_CHECK(perm_arg->get_output_element_type(0).compatible(element::i64));

        if (perm_arg->get_output_element_type(0).is_dynamic() ||
            perm_arg->get_output_partial_shape(0).is_dynamic())
        {
            return false;
        }

        auto perm = perm_arg->get_axis_vector_val();

        auto output_shape = ngraph::apply_permutation(data_shape, perm);

        auto replacement = std::make_shared<op::v0::Reshape>(data_arg, perm, output_shape);

        m.get_match_value().replace(replacement->output(0));
        return true;
    };

    auto transpose_matcher = make_shared<pattern::Matcher>(transpose, "DynElimination.Transpose");
    add_matcher(transpose_matcher, transpose_callback, all_pass_property_off);
}

void pass::DynElimination::construct_dyn_broadcast()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto shape_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());
    auto axes_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{0}, pattern::has_class<op::v0::Constant>());

    auto dyn_broadcast =
        make_shared<op::v0::DynBroadcast>(data_arg_label, shape_arg_label, axes_arg_label);

    auto dyn_broadcast_callback =
        [data_arg_label, shape_arg_label, axes_arg_label](pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_map();

            auto data_arg = m.get_pattern_value_map()[data_arg_label];
            auto shape_arg = static_pointer_cast<op::v0::Constant>(pattern_map[shape_arg_label]);
            auto axes_arg = static_pointer_cast<op::v0::Constant>(pattern_map[axes_arg_label]);

            NGRAPH_CHECK(shape_arg->get_output_partial_shape(0).rank().compatible(1));
            NGRAPH_CHECK(shape_arg->get_output_element_type(0).compatible(element::i64));
            NGRAPH_CHECK(axes_arg->get_output_partial_shape(0).rank().compatible(1));
            NGRAPH_CHECK(axes_arg->get_output_element_type(0).compatible(element::i64));

            if (shape_arg->get_output_element_type(0).is_dynamic() ||
                shape_arg->get_output_partial_shape(0).is_dynamic() ||
                axes_arg->get_output_element_type(0).is_dynamic() ||
                axes_arg->get_output_partial_shape(0).is_dynamic())
            {
                return false;
            }

            auto shape = shape_arg->get_shape_val();
            auto axes = axes_arg->get_axis_vector_val();

            auto replacement = std::make_shared<op::v0::Broadcast>(data_arg, shape, axes);

            m.get_match_value().replace(replacement->output(0));
            return true;
        };

    auto dyn_broadcast_matcher =
        make_shared<pattern::Matcher>(dyn_broadcast, "DynElimination.DynBroadcast");
    add_matcher(dyn_broadcast_matcher, dyn_broadcast_callback, all_pass_property_off);
}

void pass::DynElimination::construct_dyn_slice()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto begins_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());
    auto ends_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());
    auto strides_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());

    auto dyn_slice_pat = make_shared<op::v0::DynSlice>(data_arg_label,
                                                       begins_arg_label,
                                                       ends_arg_label,
                                                       strides_arg_label,
                                                       AxisSet{},
                                                       AxisSet{},
                                                       AxisSet{},
                                                       AxisSet{},
                                                       AxisSet{});
    auto dyn_slice_callback = [data_arg_label, begins_arg_label, ends_arg_label, strides_arg_label](
                                  pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto data_arg = m.get_pattern_value_map()[data_arg_label];
        auto begins_arg = static_pointer_cast<op::v0::Constant>(pattern_map[begins_arg_label]);
        auto ends_arg = static_pointer_cast<op::v0::Constant>(pattern_map[ends_arg_label]);
        auto strides_arg = static_pointer_cast<op::v0::Constant>(pattern_map[strides_arg_label]);
        auto dyn_slice = m.get_match_root_as<op::v0::DynSlice>();
        NGRAPH_CHECK(
            dyn_slice, "match root node ", *m.get_match_root(), " not of type `op::v0::DynSlice`");

        if (data_arg.get_partial_shape().is_dynamic() ||
            begins_arg->get_output_element_type(0) != element::i64 ||
            ends_arg->get_output_element_type(0) != element::i64 ||
            strides_arg->get_output_element_type(0) != element::i64)
        {
            return false;
        }

        SlicePlan p = make_slice_plan(data_arg.get_shape(),
                                      begins_arg->get_vector<int64_t>(),
                                      ends_arg->get_vector<int64_t>(),
                                      strides_arg->get_vector<int64_t>(),
                                      dyn_slice->get_lower_bounds_mask(),
                                      dyn_slice->get_upper_bounds_mask(),
                                      dyn_slice->get_new_axis(),
                                      dyn_slice->get_shrink_axis(),
                                      dyn_slice->get_ellipsis_mask());

        shared_ptr<Node> replacement =
            make_shared<op::v0::Slice>(data_arg,
                                       Coordinate(p.begins.begin(), p.begins.end()),
                                       Coordinate(p.ends.begin(), p.ends.end()),
                                       Strides(p.strides.begin(), p.strides.end()));

        if (p.reshape_in_shape != p.reshape_out_shape)
        {
            replacement = make_shared<op::v0::Reshape>(
                replacement, ngraph::get_default_order(p.reshape_in_shape), p.reshape_out_shape);
        }

        if (!p.reverse_axes.empty())
        {
            replacement = make_shared<op::v0::Reverse>(replacement, p.reverse_axes);
        }

        m.get_match_value().replace(replacement->output(0));
        return true;
    };

    auto dyn_slice_matcher =
        make_shared<pattern::Matcher>(dyn_slice_pat, "DynElimination.DynSlice");
    add_matcher(dyn_slice_matcher, dyn_slice_callback, all_pass_property_off);
}

void pass::DynElimination::construct_dyn_replace_slice()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto replacement_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto begins_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());
    auto ends_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());
    auto strides_arg_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::v0::Constant>());

    auto dyn_replace_slice_pat = make_shared<op::v0::DynReplaceSlice>(data_arg_label,
                                                                      replacement_arg_label,
                                                                      begins_arg_label,
                                                                      ends_arg_label,
                                                                      strides_arg_label,
                                                                      AxisSet{},
                                                                      AxisSet{},
                                                                      AxisSet{},
                                                                      AxisSet{},
                                                                      AxisSet{});
    auto dyn_replace_slice_callback = [data_arg_label,
                                       replacement_arg_label,
                                       begins_arg_label,
                                       ends_arg_label,
                                       strides_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto pvm = m.get_pattern_value_map();

        auto data_arg = pvm[data_arg_label];
        auto replacement_arg = pvm[replacement_arg_label];
        auto begins_arg = static_pointer_cast<op::v0::Constant>(pattern_map[begins_arg_label]);
        auto ends_arg = static_pointer_cast<op::v0::Constant>(pattern_map[ends_arg_label]);
        auto strides_arg = static_pointer_cast<op::v0::Constant>(pattern_map[strides_arg_label]);
        auto dyn_replace_slice = m.get_match_root_as<op::v0::DynReplaceSlice>();
        NGRAPH_CHECK(dyn_replace_slice,
                     "match root node ",
                     *m.get_match_root(),
                     " not of type `op::v0::DynReplaceSlice`");

        if (data_arg.get_partial_shape().is_dynamic() ||
            replacement_arg.get_partial_shape().is_dynamic() ||
            begins_arg->get_output_element_type(0) != element::i64 ||
            ends_arg->get_output_element_type(0) != element::i64 ||
            strides_arg->get_output_element_type(0) != element::i64)
        {
            return false;
        }

        SlicePlan p = make_slice_plan(data_arg.get_shape(),
                                      begins_arg->get_vector<int64_t>(),
                                      ends_arg->get_vector<int64_t>(),
                                      strides_arg->get_vector<int64_t>(),
                                      dyn_replace_slice->get_lower_bounds_mask(),
                                      dyn_replace_slice->get_upper_bounds_mask(),
                                      dyn_replace_slice->get_new_axis(),
                                      dyn_replace_slice->get_shrink_axis(),
                                      dyn_replace_slice->get_ellipsis_mask());

        Output<Node> substitute_replacement_arg = replacement_arg;

        if (!p.reverse_axes.empty())
        {
            substitute_replacement_arg =
                make_shared<op::v0::Reverse>(substitute_replacement_arg, p.reverse_axes);
        }

        if (p.reshape_in_shape != p.reshape_out_shape)
        {
            substitute_replacement_arg =
                make_shared<op::v0::Reshape>(substitute_replacement_arg,
                                             ngraph::get_default_order(p.reshape_out_shape),
                                             p.reshape_in_shape);
        }

        auto substitute_rsl =
            make_shared<op::v0::ReplaceSlice>(data_arg,
                                              substitute_replacement_arg,
                                              Coordinate(p.begins.begin(), p.begins.end()),
                                              Coordinate(p.ends.begin(), p.ends.end()),
                                              Strides(p.strides.begin(), p.strides.end()));

        m.get_match_value().replace(substitute_rsl->output(0));
        return true;
    };

    auto dyn_replace_slice_matcher =
        make_shared<pattern::Matcher>(dyn_replace_slice_pat, "DynElimination.DynReplaceShape");
    add_matcher(dyn_replace_slice_matcher, dyn_replace_slice_callback, all_pass_property_off);
}

template <typename T>
std::shared_ptr<op::v0::Constant>
    make_range_replacement(const element::Type& et,
                           const Shape& shape,
                           const std::shared_ptr<op::v0::Constant>& start_arg,
                           const std::shared_ptr<op::v0::Constant>& step_arg)
{
    std::vector<T> elements(shape_size(shape));
    std::vector<T> start_vec = start_arg->get_vector<T>();
    std::vector<T> step_vec = step_arg->get_vector<T>();

    NGRAPH_CHECK(start_vec.size() == 1 && step_vec.size() == 1);

    runtime::reference::range<T>(start_vec.data(), step_vec.data(), shape, elements.data());

    return make_shared<op::v0::Constant>(et, shape, elements);
}

void pass::DynElimination::construct_range()
{
    auto start_arg_label = make_shared<pattern::op::Label>(
        element::f32, Shape{}, pattern::has_class<op::v0::Constant>());
    auto stop_arg_label = make_shared<pattern::op::Label>(
        element::f32, Shape{}, pattern::has_class<op::v0::Constant>());
    auto step_arg_label = make_shared<pattern::op::Label>(
        element::f32, Shape{}, pattern::has_class<op::v0::Constant>());

    auto range_pat = make_shared<op::v0::Range>(start_arg_label, stop_arg_label, step_arg_label);

    auto range_callback = [start_arg_label, stop_arg_label, step_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto start_arg = static_pointer_cast<op::v0::Constant>(pattern_map[start_arg_label]);
        auto step_arg = static_pointer_cast<op::v0::Constant>(pattern_map[step_arg_label]);
        auto range_node = m.get_match_root_as<op::v0::Range>();
        NGRAPH_CHECK(
            range_node, "match root node ", *m.get_match_root(), " not of type `op::v0::Range`");

        NGRAPH_CHECK(start_arg->get_output_partial_shape(0).rank().compatible(0) &&
                     step_arg->get_output_partial_shape(0).rank().compatible(0));

        auto et = range_node->get_output_element_type(0);
        auto shape = range_node->get_output_shape(0);

        std::shared_ptr<op::v0::Constant> replacement;

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (et)
        {
        case element::Type_t::bf16:
            replacement = make_range_replacement<bfloat16>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f16:
            replacement = make_range_replacement<float16>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f32:
            replacement = make_range_replacement<float>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f64:
            replacement = make_range_replacement<double>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i8:
            replacement = make_range_replacement<int8_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i16:
            replacement = make_range_replacement<int16_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i32:
            replacement = make_range_replacement<int32_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i64:
            replacement = make_range_replacement<int64_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u8:
            replacement = make_range_replacement<uint8_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u16:
            replacement = make_range_replacement<uint16_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u32:
            replacement = make_range_replacement<uint32_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u64:
            replacement = make_range_replacement<uint64_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u1:
        case element::Type_t::undefined:
        case element::Type_t::dynamic:
        case element::Type_t::boolean:
            NGRAPH_CHECK(false, "Internal nGraph error: unsupported element type: ", et);
            break;
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

        replace_node(range_node, replacement);
        return true;
    };

    auto range_matcher = make_shared<pattern::Matcher>(range_pat, "DynElimination.Range");
    add_matcher(range_matcher, range_callback, all_pass_property_off);
}
