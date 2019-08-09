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
#include "ngraph/op/select.hpp"
#include "ngraph/runtime/reference/select.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_select(shared_ptr<op::Constant> selection,
                                              shared_ptr<op::Constant> t,
                                              shared_ptr<op::Constant> f,
                                              shared_ptr<op::Select> select)
{
    auto out_shape = select->get_shape();
    vector<T> out_vec(shape_size(out_shape));

    runtime::reference::select<T>(selection->get_data_ptr<char>(),
                                  t->get_data_ptr<T>(),
                                  f->get_data_ptr<T>(),
                                  out_vec.data(),
                                  shape_size(out_shape));

    return make_shared<op::Constant>(select->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_select()
{
    auto selection_label = make_shared<pattern::op::Label>(
        element::boolean, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto t_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto f_label = make_shared<pattern::op::Label>(
        element::i64, Shape{2, 3, 4}, pattern::has_class<op::Constant>());
    auto select_op = make_shared<op::Select>(selection_label, t_label, f_label);

    auto constant_select_callback = [selection_label, t_label, f_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_select_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto selection_node = static_pointer_cast<op::Constant>(pattern_map[selection_label]);
        auto t_node = static_pointer_cast<op::Constant>(pattern_map[t_label]);
        auto f_node = static_pointer_cast<op::Constant>(pattern_map[f_label]);
        auto select = static_pointer_cast<op::Select>(m.get_match_root());

        NGRAPH_CHECK(revalidate_and_ensure_static(select));

        std::shared_ptr<op::Constant> replacement;

        switch (select->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in constant_select_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in constant_select_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_select<char>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_select<bfloat16>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_select<float16>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_select<float>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_select<double>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_select<int8_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_select<int16_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_select<int32_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_select<int64_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_select<uint8_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_select<uint16_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_select<uint32_t>(selection_node, t_node, f_node, select);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_select<uint64_t>(selection_node, t_node, f_node, select);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto select_matcher =
        make_shared<pattern::Matcher>(select_op, "ConstantFolding.ConstantSelect");
    this->add_matcher(select_matcher, constant_select_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
