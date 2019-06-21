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

#include "dyn_elimination.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/label.hpp"

using namespace std;
using namespace ngraph;

pass::DynElimination::DynElimination()
    : GraphRewrite()
{
    construct_transpose();
    construct_broadcast();
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

void pass::DynElimination::construct_broadcast()
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

//
// We eliminate DynSlice by converting it to a sequence of ops:
//
//      Slice    (to do the basic slicing)
//        |
//        v
//     Reshape   (non-transposing, to handle shrinks)
//        |
//        v
//     Reverse   (to emulate backwards stride)
//
// (The Reshape, Reverse, or both may be omitted if they would just be identities.)
//
// A SlicePlan is used to collect parameters for these ops.
//
struct SlicePlan
{
    // Parameters for the Slice
    std::vector<int64_t> begins;
    std::vector<int64_t> ends;
    std::vector<int64_t> strides;

    // Shapes coming into, and going out of, the Reshape.
    Shape reshape_in_shape;
    Shape reshape_out_shape;

    // Parameters for the Reverse
    std::set<size_t> reverse_axes;
};

static SlicePlan make_plan(const Shape& input_shape,
                           const std::vector<int64_t>& begins,
                           const std::vector<int64_t>& ends,
                           const std::vector<int64_t>& strides,
                           const AxisSet& lower_bounds_mask,
                           const AxisSet& upper_bounds_mask,
                           const AxisSet& new_axis_mask,
                           const AxisSet& shrink_axis_mask,
                           const AxisSet& ellipsis_mask)
{
    NGRAPH_CHECK(begins.size() == ends.size());
    NGRAPH_CHECK(ends.size() == strides.size());
    size_t num_slice_indices = begins.size();

    size_t num_real_axes = 0;
    size_t num_shrink_axes = 0;
    size_t num_new_axes = 0;
    bool ellipsis_found = false;

    // Make a pass over the original slices to make sure there is at most one
    // ellipsis, and to count up the number of shrink axes, the number of
    // "newaxis"es, and the number of "real" axes (axes that are not newaxis
    // and are not the ellipsis).
    for (size_t i = 0; i < num_slice_indices; i++)
    {
        if (ellipsis_mask.count(i))
        {
            NGRAPH_CHECK(!ellipsis_found);
            ellipsis_found = true;
        }
        else if (new_axis_mask.count(i))
        {
            num_new_axes++;
        }
        else
        {
            if (shrink_axis_mask.count(i))
            {
                num_shrink_axes++;
            }
            num_real_axes++;
        }
    }

    NGRAPH_CHECK(num_real_axes <= input_shape.size(),
                 "num_real_axes=",
                 num_real_axes,
                 ", input_shape=",
                 input_shape);

    // Figure out how many axes need to be inserted when the ellipsis (which
    // may be an implicit ellipsis at the end) is expanded.
    size_t ellipsis_size = input_shape.size() - num_real_axes;

    // Initialize our slice plan.
    SlicePlan p;
    p.begins = std::vector<int64_t>(num_real_axes + ellipsis_size);
    p.ends = std::vector<int64_t>(num_real_axes + ellipsis_size);
    p.strides = std::vector<int64_t>(num_real_axes + ellipsis_size);
    p.reshape_in_shape = Shape(num_real_axes + ellipsis_size);
    p.reshape_out_shape = Shape(num_new_axes + num_real_axes + ellipsis_size - num_shrink_axes);
    p.reverse_axes = AxisSet{};

    // Begin a maddeningly delicate loop to desugar the original slice.
    //
    // * i_in is iterating over the axes of the input shape, which are also the axes of
    //     p.reshape_in_shape.
    // * i_out is iterating over the axes of p.reshape_out_shape
    size_t i_in = 0;
    size_t i_out = 0;

    // If no actual ellipsis exists, there is an "implicit" one at the end,
    // which we will handle after the loop. So the logic is wrapped up here,
    // allowing it to be used both during and after the loop.
    auto expand_ellipsis = [&]() {
        for (size_t i = 0; i < ellipsis_size; i++)
        {
            p.begins[i_in] = 0;
            p.ends[i_in] = int64_t(input_shape[i_in]);
            p.strides[i_in] = 1;
            p.reshape_in_shape[i_in] = input_shape[i_in];
            p.reshape_out_shape[i_out] = input_shape[i_in];

            i_in++;
            i_out++;
        }
    };

    for (size_t i = 0; i < num_slice_indices; i++)
    {
        // If this is a "newaxis", then reshape_out_shape will have a 1 here,
        // but reshape_in_shape will not.
        if (new_axis_mask.count(i))
        {
            p.reshape_out_shape[i_out] = 1;
            i_out++;
        }
        // If this is a "shrunken" axis, then reshape_in_shape will have a 1
        // here, but reshape_out_shape will not.
        else if (shrink_axis_mask.count(i))
        {
            int64_t begin = begins[i];

            // Note that clipping is not used for "shrunken" axes: an
            // out-of-bounds index is an error.
            NGRAPH_CHECK(begin >= -(int64_t(input_shape[i_in])) &&
                         begin < int64_t(input_shape[i_in]));

            if (begin < 0)
            {
                begin += int64_t(input_shape[i_in]);
            }
            p.begins[i_in] = begin;
            p.ends[i_in] = begin + 1;
            p.strides[i_in] = 1;
            p.reshape_in_shape[i_in] = 1;
            i_in++;
        }
        // If this is the ellipsis, expand it.
        else if (ellipsis_mask.count(i))
        {
            expand_ellipsis();
        }
        // In other cases, we have a nice, ordinary (begin:end:stride) slice.
        // We need to adjust for begin/end being masked, and begin/end/stride
        // being negative or out of bounds.
        else
        {
            bool is_reverse = strides[i] < 0;

            // Adjust the beginning for from-the-right indexing, and clip.
            int64_t real_begin = begins[i];
            if (lower_bounds_mask.count(i))
            {
                real_begin = (is_reverse ? int64_t(input_shape[i_in] - 1) : 0);
            }
            else if (real_begin < 0)
            {
                real_begin += int64_t(input_shape[i_in]);
            }
            int64_t max_real_begin = int64_t(input_shape[i_in]) - (is_reverse ? 1 : 0);
            real_begin = std::max(int64_t(0), std::min(max_real_begin, real_begin));

            // Adjust the ending for from-the-right indexing, and clip.
            int64_t real_end = ends[i];
            if (upper_bounds_mask.count(i))
            {
                real_end = (is_reverse ? -1 : int64_t(input_shape[i_in]));
            }
            else if (real_end < 0)
            {
                real_end += int64_t(input_shape[i_in]);
            }
            int64_t min_real_end = (is_reverse ? -1 : 0);
            real_end = std::max(min_real_end, std::min(int64_t(input_shape[i_in]), real_end));

            // Ensure stride is not zero, and adjust it for backwards slicing.
            NGRAPH_CHECK(strides[i] != 0);
            int64_t real_stride = std::abs(strides[i]);

            // Adjust for reversal if needed. This isn't quite as simple as swapping begin and
            // end, due to striding; we have to adjust the end point to be the _actual_ leftmost
            // element, in cases where the stride does not evenly divide the span between begin
            // and end.
            if (is_reverse)
            {
                real_end += std::max(int64_t(0), real_begin - real_end - 1) % real_stride;
                std::swap(real_begin, real_end);
                real_begin++;
                real_end++;
                p.reverse_axes.insert(i_out);
            }

            // nGraph's slice op does not like it when end < begin, so we truncate for that case
            // here.
            if (real_end < real_begin)
            {
                real_end = real_begin;
            }

            // Compute output dimension.
            size_t dim = (real_end <= real_begin
                              ? 0
                              : size_t(real_end - real_begin - 1) / size_t(real_stride) + 1);
            p.reshape_in_shape[i_in] = dim;
            p.reshape_out_shape[i_out] = dim;

            // Set up the begin/end/stride.
            p.begins[i_in] = real_begin;
            p.ends[i_in] = real_end;
            p.strides[i_in] = real_stride;

            i_in++;
            i_out++;
        }
    }

    // If there was no ellipsis explicitly given, there is an implicit one at
    // the end (it might encompass zero axes, but that's fine).
    if (!ellipsis_found)
    {
        expand_ellipsis();
    }
    return p;
}

void pass::DynElimination::construct_dyn_reshape()
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

        SlicePlan p = make_plan(data_arg->get_output_shape(0),
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
        make_shared<pattern::Matcher>(dyn_slice_pat, "DynElimination.DynShape");
    add_matcher(dyn_slice_matcher, dyn_slice_callback, all_pass_property_off);
}

template <typename T>
std::shared_ptr<op::Constant>
    make_range_replacement_integral(const element::Type& et,
                                    const Shape& shape,
                                    const std::shared_ptr<op::Constant>& start_arg,
                                    const std::shared_ptr<op::Constant>& step_arg)
{
    std::vector<T> elements(shape_size(shape));
    std::vector<T> start_vec = start_arg->get_vector<T>();
    std::vector<T> step_vec = step_arg->get_vector<T>();

    NGRAPH_CHECK(start_vec.size() == 1 && step_vec.size() == 1);

    T start = start_vec[0];
    T step = step_vec[0];

    T val = start;

    for (size_t i = 0; i < elements.size(); i++)
    {
        elements[i] = val;
        val = val + step;
    }

    return make_shared<op::Constant>(et, shape, elements);
}

template <typename T>
std::shared_ptr<op::Constant>
    make_range_replacement_floating(const element::Type& et,
                                    const Shape& shape,
                                    const std::shared_ptr<op::Constant>& start_arg,
                                    const std::shared_ptr<op::Constant>& step_arg)
{
    std::vector<T> elements(shape_size(shape));
    std::vector<T> start_vec = start_arg->get_vector<T>();
    std::vector<T> step_vec = step_arg->get_vector<T>();

    NGRAPH_CHECK(start_vec.size() == 1 && step_vec.size() == 1);

    T start = start_vec[0];
    T step = step_vec[0];

    for (size_t i = 0; i < elements.size(); i++)
    {
        elements[i] = start + (static_cast<T>(i) * step);
    }

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

#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
        switch (et.get_type_enum())
        {
        case element::Type_t::bf16:
            replacement = make_range_replacement_floating<bfloat16>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f16:
            replacement = make_range_replacement_floating<float16>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f32:
            replacement = make_range_replacement_floating<float>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::f64:
            replacement = make_range_replacement_floating<double>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i8:
            replacement = make_range_replacement_integral<int8_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i16:
            replacement = make_range_replacement_integral<int16_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i32:
            replacement = make_range_replacement_integral<int32_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::i64:
            replacement = make_range_replacement_integral<int64_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u8:
            replacement = make_range_replacement_integral<uint8_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u16:
            replacement = make_range_replacement_integral<uint16_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u32:
            replacement = make_range_replacement_integral<uint32_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::u64:
            replacement = make_range_replacement_integral<uint64_t>(et, shape, start_arg, step_arg);
            break;
        case element::Type_t::undefined:
        case element::Type_t::dynamic:
        case element::Type_t::boolean:
            NGRAPH_CHECK(false, "Internal nGraph error: unsupported element type: ", et);
            break;
        }
#if !(defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

        replace_node(range_node, replacement);
        return true;
    };

    auto range_matcher = make_shared<pattern::Matcher>(range_pat, "DynElimination.Range");
    add_matcher(range_matcher, range_callback, all_pass_property_off);
}
