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

#include "constant_folding.hpp"
#include "ngraph/op/strided_slice.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/runtime/reference/reverse.hpp"
#include "ngraph/runtime/reference/slice.hpp"
#include "ngraph/slice_plan.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_strided_slice(shared_ptr<op::Constant> data,
                                                     shared_ptr<op::Constant> begin,
                                                     shared_ptr<op::Constant> end,
                                                     shared_ptr<op::Constant> strides,
                                                     shared_ptr<op::v1::StridedSlice> slice)
{
    auto convert_mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        AxisSet axis_set{};
        for (size_t i = 0; i < static_cast<size_t>(mask.size()); ++i)
        {
            if (mask[i] == 1)
            {
                axis_set.emplace(i);
            }
        }
        return axis_set;
    };

    SlicePlan plan = make_slice_plan(data->get_shape(),
                                     begin->get_vector<int64_t>(),
                                     end->get_vector<int64_t>(),
                                     strides->get_vector<int64_t>(),
                                     convert_mask_to_axis_set(slice->get_begin_mask()),
                                     convert_mask_to_axis_set(slice->get_end_mask()),
                                     convert_mask_to_axis_set(slice->get_new_axis_mask()),
                                     convert_mask_to_axis_set(slice->get_shrink_axis_mask()),
                                     convert_mask_to_axis_set(slice->get_ellipsis_mask()));

    runtime::AlignedBuffer slice_out_buffer(shape_size(plan.reshape_in_shape) * sizeof(T));
    runtime::reference::slice<T>(data->get_data_ptr<T>(),
                                 slice_out_buffer.get_ptr<T>(),
                                 data->get_shape(),
                                 Coordinate(plan.begins.begin(), plan.begins.end()),
                                 Coordinate(plan.ends.begin(), plan.ends.end()),
                                 Strides(plan.strides.begin(), plan.strides.end()),
                                 plan.reshape_in_shape);

    runtime::AlignedBuffer reshape_out_buffer(shape_size(plan.reshape_out_shape) * sizeof(T));
    runtime::reference::reshape<T>(slice_out_buffer.get_ptr<T>(),
                                   reshape_out_buffer.get_ptr<T>(),
                                   plan.reshape_in_shape,
                                   get_default_order(plan.reshape_in_shape.size()),
                                   plan.reshape_out_shape);

    runtime::AlignedBuffer reverse_out_buffer(shape_size(plan.reshape_out_shape) * sizeof(T));
    runtime::reference::reverse<T>(reshape_out_buffer.get_ptr<T>(),
                                   reverse_out_buffer.get_ptr<T>(),
                                   plan.reshape_out_shape,
                                   plan.reshape_out_shape,
                                   plan.reverse_axes);

    return make_shared<op::Constant>(
        data->get_element_type(), plan.reshape_out_shape, reverse_out_buffer.get_ptr<T>());
}

void pass::ConstantFolding::construct_constant_strided_slice()
{
    auto data_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto begin_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto end_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto strides_label =
        make_shared<pattern::op::Label>(element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto strided_slice_op = make_shared<op::v1::StridedSlice>(data_label,
                                                              begin_label,
                                                              end_label,
                                                              strides_label,
                                                              std::vector<int64_t>{},
                                                              std::vector<int64_t>{},
                                                              std::vector<int64_t>{},
                                                              std::vector<int64_t>{},
                                                              std::vector<int64_t>{});

    auto constant_strided_slice_callback =
        [data_label, begin_label, end_label, strides_label](pattern::Matcher& m) {
            NGRAPH_DEBUG << "In callback for constant_strided_slice_callback against node = "
                         << m.get_match_root()->get_name();

            auto pattern_map = m.get_pattern_map();

            auto data_node = static_pointer_cast<op::Constant>(pattern_map[data_label]);
            auto begin_node = static_pointer_cast<op::Constant>(pattern_map[begin_label]);
            auto end_node = static_pointer_cast<op::Constant>(pattern_map[end_label]);
            auto strides_node = static_pointer_cast<op::Constant>(pattern_map[strides_label]);
            auto strided_slice = static_pointer_cast<op::v1::StridedSlice>(m.get_match_root());

            NGRAPH_CHECK(revalidate_and_ensure_static(strided_slice));

            std::shared_ptr<op::Constant> replacement;

            switch (strided_slice->get_output_element_type(0))
            {
            case element::Type_t::undefined:
                NGRAPH_CHECK(false,
                             "Encountered 'undefined' element type in fold_constant_strided_slice");
                break;
            case element::Type_t::dynamic:
                NGRAPH_CHECK(false,
                             "Encountered 'dynamic' element type in fold_constant_strided_slice");
                break;
            case element::Type_t::u1:
                NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_strided_slice");
                break;
            case element::Type_t::boolean:
                replacement = fold_constant_strided_slice<char>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::bf16:
                replacement = fold_constant_strided_slice<bfloat16>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::f16:
                replacement = fold_constant_strided_slice<float16>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::f32:
                replacement = fold_constant_strided_slice<float>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::f64:
                replacement = fold_constant_strided_slice<double>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::i8:
                replacement = fold_constant_strided_slice<int8_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::i16:
                replacement = fold_constant_strided_slice<int16_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::i32:
                replacement = fold_constant_strided_slice<int32_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::i64:
                replacement = fold_constant_strided_slice<int64_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::u8:
                replacement = fold_constant_strided_slice<uint8_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::u16:
                replacement = fold_constant_strided_slice<uint16_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::u32:
                replacement = fold_constant_strided_slice<uint32_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            case element::Type_t::u64:
                replacement = fold_constant_strided_slice<uint64_t>(
                    data_node, begin_node, end_node, strides_node, strided_slice);
                break;
            }

            replace_node(m.get_match_root(), replacement);
            return true;
        };

    auto strided_slice_matcher =
        make_shared<pattern::Matcher>(strided_slice_op, "ConstantFolding.ConstantStridedSlice");
    this->add_matcher(
        strided_slice_matcher, constant_strided_slice_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
