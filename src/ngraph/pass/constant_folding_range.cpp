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

#include "constant_folding.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/runtime/reference/range.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_range(shared_ptr<op::Constant> start,
                                             shared_ptr<op::Constant> step,
                                             shared_ptr<op::Range> range)
{
    runtime::AlignedBuffer buffer(shape_size(range->get_shape()) * sizeof(T));
    T* data_ptr = buffer.get_ptr<T>();
    runtime::reference::range<T>(
        start->get_vector<T>().data(), step->get_vector<T>().data(), range->get_shape(), data_ptr);

    return make_shared<op::Constant>(range->get_element_type(), range->get_shape(), data_ptr);
}

void pass::ConstantFolding::construct_constant_range()
{
    auto start_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto stop_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto step_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto range_op = make_shared<op::Range>(start_label, stop_label, step_label);

    auto constant_range_callback = [start_label, stop_label, step_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_range_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto start_node = static_pointer_cast<op::Constant>(pattern_map[start_label]);
        auto stop_node = static_pointer_cast<op::Constant>(pattern_map[stop_label]);
        auto step_node = static_pointer_cast<op::Constant>(pattern_map[step_label]);
        auto range = static_pointer_cast<op::Range>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(range));

        std::shared_ptr<op::Constant> replacement;

        switch (range->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_range_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_range_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in constant_range_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_range<char>(start_node, step_node, range);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_range<bfloat16>(start_node, step_node, range);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_range<float16>(start_node, step_node, range);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_range<float>(start_node, step_node, range);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_range<double>(start_node, step_node, range);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_range<int8_t>(start_node, step_node, range);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_range<int16_t>(start_node, step_node, range);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_range<int32_t>(start_node, step_node, range);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_range<int64_t>(start_node, step_node, range);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_range<uint8_t>(start_node, step_node, range);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_range<uint16_t>(start_node, step_node, range);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_range<uint32_t>(start_node, step_node, range);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_range<uint64_t>(start_node, step_node, range);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto range_matcher = make_shared<pattern::Matcher>(range_op, "ConstantFolding.ConstantRange");
    this->add_matcher(range_matcher, constant_range_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
