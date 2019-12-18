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
#include "ngraph/op/constant.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/runtime/reference/one_hot.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_one_hot(const shared_ptr<op::Constant>& indices, const shared_ptr<Node>& one_hot)
{
    if (auto one_hot_v0 = as_type_ptr<op::v0::OneHot>(one_hot))
    {
        std::vector<OUTPUT_TYPE> out_vec(shape_size(one_hot_v0->get_shape()));
        runtime::reference::one_hot<OUTPUT_TYPE>(indices->get_data_ptr<T>(),
                                    out_vec.data(),
                                    indices->get_shape(),
                                    one_hot_v0->get_output_shape(0),
                                    one_hot_v0->get_one_hot_axis());

        return make_shared<op::Constant>(one_hot_v0->get_output_element_type(0), one_hot_v0->get_output_shape(0), out_vec);
    }
    else
    {
        throw ngraph_error("Unsupported op in one_hot constant folding.");
    }
}

void pass::ConstantFolding::construct_constant_one_hot()
{
    auto indices_label = make_shared<pattern::op::Label>(
        element::i64, Shape{3}, pattern::has_class<op::Constant>());
    auto depth_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto on_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    auto off_label =
        make_shared<pattern::op::Label>(element::i64, Shape{}, pattern::has_class<op::Constant>());
    int64_t axis = 0;
    auto ont_hot_pattern = make_shared<op::v1::OneHot>(indices_label, depth_label, on_label, off_label, axis);

    auto one_hot_callback = [indices_label, depth_label, on_label, off_label](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for one_hot_callback against node = "
                     << m.get_match_root()->get_name();
        auto pattern_map = m.get_pattern_map();

        auto indices_node = static_pointer_cast<op::Constant>(pattern_map[indices_label]);
        const auto depth_node = static_pointer_cast<op::Constant>(pattern_map[depth_label]);
        const auto on_node = static_pointer_cast<op::Constant>(pattern_map[on_label]);
        const auto off_node = static_pointer_cast<op::Constant>(pattern_map[off_label]);

        auto one_hot = static_pointer_cast<op::v1::OneHot>(m.get_match_root());
        Opset0Downgrade downgrade_pass;
        downgrade_pass.run_on_node(one_hot);
        
        std::shared_ptr<Node> replacement;
        auto output_type = on_node->get_element_type();
        /*
        switch (output_type)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in one_hot_callback");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in one_hot_callback");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in one_hot_callback");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_one_hot<char>(indices_node, one_hot);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_one_hot<bfloat16>(indices_node, one_hot);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_one_hot<float16>(indices_node, one_hot);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_one_hot<float>(indices_node, one_hot);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_one_hot<double>(indices_node, one_hot);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_one_hot<int8_t>(indices_node, one_hot);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_one_hot<int16_t>(indices_node, one_hot);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_one_hot<int32_t>(indices_node, one_hot);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_one_hot<int64_t>(indices_node, one_hot);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_one_hot<uint8_t>(indices_node, one_hot);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_one_hot<uint16_t>(indices_node, one_hot);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_one_hot<uint32_t>(indices_node, one_hot);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_one_hot<uint64_t>(indices_node, one_hot);
            break;
        }*/

        replace_node(m.get_match_root(), replacement);
        return true;
    };
    auto split_matcher =
        make_shared<pattern::Matcher>(ont_hot_pattern, "ConstantFolding.ConstantOneHot");
    this->add_matcher(split_matcher, one_hot_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
