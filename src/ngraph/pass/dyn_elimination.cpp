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
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
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
    construct_dyn_reshape();
    construct_range();
}

void pass::DynElimination::construct_transpose()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto perm_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());

    auto transpose = make_shared<op::Transpose>(data_arg_label, perm_arg_label);

    auto transpose_callback = [data_arg_label, perm_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto data_arg = pattern_map[data_arg_label];
        auto perm_arg = static_pointer_cast<op::Constant>(pattern_map[perm_arg_label]);

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

        auto replacement = std::make_shared<op::Reshape>(data_arg, perm, output_shape);

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto transpose_matcher = make_shared<pattern::Matcher>(transpose, "DynElimination.Transpose");
    add_matcher(transpose_matcher, transpose_callback, all_pass_property_off);
}

void pass::DynElimination::construct_dyn_broadcast()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto shape_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto axes_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{0}, pattern::has_class<op::Constant>());

    auto dyn_broadcast =
        make_shared<op::DynBroadcast>(data_arg_label, shape_arg_label, axes_arg_label);

    auto dyn_broadcast_callback =
        [data_arg_label, shape_arg_label, axes_arg_label](pattern::Matcher& m) {
            auto pattern_map = m.get_pattern_map();

            auto data_arg = pattern_map[data_arg_label];
            auto shape_arg = static_pointer_cast<op::Constant>(pattern_map[shape_arg_label]);
            auto axes_arg = static_pointer_cast<op::Constant>(pattern_map[axes_arg_label]);

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

            auto replacement = std::make_shared<op::Broadcast>(data_arg, shape, axes);

            replace_node(m.get_match_root(), replacement);
            return true;
        };

    auto dyn_broadcast_matcher =
        make_shared<pattern::Matcher>(dyn_broadcast, "DynElimination.DynBroadcast");
    add_matcher(dyn_broadcast_matcher, dyn_broadcast_callback, all_pass_property_off);
}

void pass::DynElimination::construct_dyn_slice()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto begins_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto ends_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto strides_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());

    auto dyn_slice_pat = make_shared<op::DynSlice>(data_arg_label,
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

        auto data_arg = pattern_map[data_arg_label];
        auto begins_arg = static_pointer_cast<op::Constant>(pattern_map[begins_arg_label]);
        auto ends_arg = static_pointer_cast<op::Constant>(pattern_map[ends_arg_label]);
        auto strides_arg = static_pointer_cast<op::Constant>(pattern_map[strides_arg_label]);
        auto dyn_slice = static_pointer_cast<op::DynSlice>(m.get_match_root());

        if (data_arg->get_output_partial_shape(0).is_dynamic() ||
            begins_arg->get_element_type() != element::i64 ||
            ends_arg->get_element_type() != element::i64 ||
            strides_arg->get_element_type() != element::i64)
        {
            return false;
        }

        SlicePlan p = make_slice_plan(data_arg->get_output_shape(0),
                                      begins_arg->get_vector<int64_t>(),
                                      ends_arg->get_vector<int64_t>(),
                                      strides_arg->get_vector<int64_t>(),
                                      dyn_slice->get_lower_bounds_mask(),
                                      dyn_slice->get_upper_bounds_mask(),
                                      dyn_slice->get_new_axis(),
                                      dyn_slice->get_shrink_axis(),
                                      dyn_slice->get_ellipsis_mask());

        shared_ptr<Node> replacement =
            make_shared<op::Slice>(data_arg,
                                   Coordinate(p.begins.begin(), p.begins.end()),
                                   Coordinate(p.ends.begin(), p.ends.end()),
                                   Strides(p.strides.begin(), p.strides.end()));

        if (p.reshape_in_shape != p.reshape_out_shape)
        {
            replacement = make_shared<op::Reshape>(
                replacement, ngraph::get_default_order(p.reshape_in_shape), p.reshape_out_shape);
        }

        if (!p.reverse_axes.empty())
        {
            replacement = make_shared<op::Reverse>(replacement, p.reverse_axes);
        }

        replace_node(m.get_match_root(), replacement);
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
    auto begins_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto ends_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto strides_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());

    auto dyn_replace_slice_pat = make_shared<op::DynReplaceSlice>(data_arg_label,
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

        auto data_arg = pattern_map[data_arg_label];
        auto replacement_arg = pattern_map[replacement_arg_label];
        auto begins_arg = static_pointer_cast<op::Constant>(pattern_map[begins_arg_label]);
        auto ends_arg = static_pointer_cast<op::Constant>(pattern_map[ends_arg_label]);
        auto strides_arg = static_pointer_cast<op::Constant>(pattern_map[strides_arg_label]);
        auto dyn_replace_slice = static_pointer_cast<op::DynReplaceSlice>(m.get_match_root());

        if (data_arg->get_output_partial_shape(0).is_dynamic() ||
            replacement_arg->get_output_partial_shape(0).is_dynamic() ||
            begins_arg->get_element_type() != element::i64 ||
            ends_arg->get_element_type() != element::i64 ||
            strides_arg->get_element_type() != element::i64)
        {
            return false;
        }

        SlicePlan p = make_slice_plan(data_arg->get_output_shape(0),
                                      begins_arg->get_vector<int64_t>(),
                                      ends_arg->get_vector<int64_t>(),
                                      strides_arg->get_vector<int64_t>(),
                                      dyn_replace_slice->get_lower_bounds_mask(),
                                      dyn_replace_slice->get_upper_bounds_mask(),
                                      dyn_replace_slice->get_new_axis(),
                                      dyn_replace_slice->get_shrink_axis(),
                                      dyn_replace_slice->get_ellipsis_mask());

        shared_ptr<Node> substitute_replacement_arg = replacement_arg;

        if (!p.reverse_axes.empty())
        {
            substitute_replacement_arg =
                make_shared<op::Reverse>(substitute_replacement_arg, p.reverse_axes);
        }

        if (p.reshape_in_shape != p.reshape_out_shape)
        {
            substitute_replacement_arg =
                make_shared<op::Reshape>(substitute_replacement_arg,
                                         ngraph::get_default_order(p.reshape_out_shape),
                                         p.reshape_in_shape);
        }

        auto substitute_rsl =
            make_shared<op::ReplaceSlice>(data_arg,
                                          substitute_replacement_arg,
                                          Coordinate(p.begins.begin(), p.begins.end()),
                                          Coordinate(p.ends.begin(), p.ends.end()),
                                          Strides(p.strides.begin(), p.strides.end()));

        replace_node(m.get_match_root(), substitute_rsl);
        return true;
    };

    auto dyn_replace_slice_matcher =
        make_shared<pattern::Matcher>(dyn_replace_slice_pat, "DynElimination.DynReplaceShape");
    add_matcher(dyn_replace_slice_matcher, dyn_replace_slice_callback, all_pass_property_off);
}

void pass::DynElimination::construct_dyn_reshape()
{
    auto data_arg_label = make_shared<pattern::op::Label>(element::f32, Shape{1, 2, 3});
    auto shape_arg_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());

    auto dyn_reshape = make_shared<op::DynReshape>(data_arg_label, shape_arg_label);

    auto dyn_reshape_callback = [data_arg_label, shape_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto data_arg = pattern_map[data_arg_label];
        auto shape_arg = static_pointer_cast<op::Constant>(pattern_map[shape_arg_label]);
        auto dyn_reshape_node = static_pointer_cast<op::DynReshape>(m.get_match_root());

        // TODO(amprocte): Can't handle the case where data rank is dynamic even if we know the
        // output shape, because static Reshape requries an axis permutation (here an identity) to
        // be given. See if we can come up with a workaround.
        if (data_arg->get_output_partial_shape(0).rank().is_dynamic())
        {
            return false;
        }

        if (dyn_reshape_node->get_output_partial_shape(0).is_dynamic())
        {
            return false;
        }

        auto& result_shape = dyn_reshape_node->get_output_shape(0);
        AxisVector perm(data_arg->get_output_partial_shape(0).rank().get_length());
        std::iota(perm.begin(), perm.end(), 0);

        auto replacement = std::make_shared<op::Reshape>(data_arg, perm, result_shape);

        replace_node(dyn_reshape_node, replacement);
        return true;
    };

    auto dyn_reshape_matcher =
        make_shared<pattern::Matcher>(dyn_reshape, "DynElimination.DynReshape");
    add_matcher(dyn_reshape_matcher, dyn_reshape_callback, all_pass_property_off);
}

template <typename T>
std::shared_ptr<op::Constant> make_range_replacement(const element::Type& et,
                                                     const Shape& shape,
                                                     const std::shared_ptr<op::Constant>& start_arg,
                                                     const std::shared_ptr<op::Constant>& step_arg)
{
    std::vector<T> elements(shape_size(shape));
    std::vector<T> start_vec = start_arg->get_vector<T>();
    std::vector<T> step_vec = step_arg->get_vector<T>();

    NGRAPH_CHECK(start_vec.size() == 1 && step_vec.size() == 1);

    runtime::reference::range<T>(start_vec.data(), step_vec.data(), shape, elements.data());

    return make_shared<op::Constant>(et, shape, elements);
}

void pass::DynElimination::construct_range()
{
    auto start_arg_label =
        make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<op::Constant>());
    auto stop_arg_label =
        make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<op::Constant>());
    auto step_arg_label =
        make_shared<pattern::op::Label>(element::f32, Shape{}, pattern::has_class<op::Constant>());

    auto range_pat = make_shared<op::Range>(start_arg_label, stop_arg_label, step_arg_label);

    auto range_callback = [start_arg_label, stop_arg_label, step_arg_label](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();

        auto start_arg = static_pointer_cast<op::Constant>(pattern_map[start_arg_label]);
        auto step_arg = static_pointer_cast<op::Constant>(pattern_map[step_arg_label]);
        auto range_node = static_pointer_cast<op::Range>(m.get_match_root());

        NGRAPH_CHECK(start_arg->get_output_partial_shape(0).rank().compatible(0) &&
                     step_arg->get_output_partial_shape(0).rank().compatible(0));

        auto et = range_node->get_output_element_type(0);
        auto shape = range_node->get_output_shape(0);

        std::shared_ptr<op::Constant> replacement;

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
