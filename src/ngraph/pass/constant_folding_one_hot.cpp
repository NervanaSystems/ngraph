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
#include "ngraph/runtime/reference/one_hot.hpp"

using namespace std;
using namespace ngraph;

template <class T>
shared_ptr<op::Constant> fold_constant_one_hot(const shared_ptr<op::Constant>& indices, const Shape& output_shape, size_t axis)
{
    std::vector<T> out_vec(shape_size(output_shape));
    runtime::reference::one_hot<T>(indices->get_data_ptr<T>(),
                                out_vec.data(),
                                indices->get_shape(),
                                output_shape,
                                axis);

    return make_shared<op::Constant>(indices->get_output_element_type(0), output_shape, out_vec);
}

shared_ptr<op::Constant> cast_constant(shared_ptr<op::Constant> constant,
                                                       const element::Type& output_et)
{
        const auto shape = constant->get_shape();
        switch (output_et)
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false,
                         "Encountered 'undefined' element type in one_hot_callback during cast_constant");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in one_hot_callback during cast_constant");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in one_hot_callback during cast_constant");
            break;
        case element::Type_t::boolean:
            return op::Constant::create(output_et, shape, constant->cast_vector<char>());
            break;
        case element::Type_t::bf16:
            return op::Constant::create(output_et, shape, constant->cast_vector<bfloat16>());
            break;
        case element::Type_t::f16:
            return op::Constant::create(output_et, shape, constant->cast_vector<float16>());
            break;
        case element::Type_t::f32:
            return op::Constant::create(output_et, shape, constant->cast_vector<float>());
            break;
        case element::Type_t::f64:
            return op::Constant::create(output_et, shape, constant->cast_vector<double>());
            break;
        case element::Type_t::i8:
            return op::Constant::create(output_et, shape, constant->cast_vector<int8_t>());
            break;
        case element::Type_t::i16:
            return op::Constant::create(output_et, shape, constant->cast_vector<int16_t>());
            break;
        case element::Type_t::i32:
            return op::Constant::create(output_et, shape, constant->cast_vector<int32_t>());
            break;
        case element::Type_t::i64:
            return op::Constant::create(output_et, shape, constant->cast_vector<int64_t>());
            break;
        case element::Type_t::u8:
            return op::Constant::create(output_et, shape, constant->cast_vector<uint16_t>());
            break;
        case element::Type_t::u16:
            return op::Constant::create(output_et, shape, constant->cast_vector<uint32_t>());
            break;
        case element::Type_t::u32:
            return op::Constant::create(output_et, shape, constant->cast_vector<char>());
            break;
        case element::Type_t::u64:
            return op::Constant::create(output_et, shape, constant->cast_vector<uint64_t>());
            break;
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
        
        auto one_hot_V1 = static_pointer_cast<op::v1::OneHot>(m.get_match_root());
        const auto output_shape = one_hot_V1->get_output_shape(0);
        const auto axis = one_hot_V1->get_axis();
        auto output_type = on_node->get_element_type();

        std::shared_ptr<op::Constant> one_hot_v0;
        switch (indices_node->get_element_type())
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
            one_hot_v0 = fold_constant_one_hot<char>(indices_node, output_shape, axis);
            break;
        case element::Type_t::bf16:
            one_hot_v0 = fold_constant_one_hot<bfloat16>(indices_node, output_shape, axis);
            break;
        case element::Type_t::f16:
            one_hot_v0 = fold_constant_one_hot<float16>(indices_node, output_shape, axis);
            break;
        case element::Type_t::f32:
            one_hot_v0 = fold_constant_one_hot<float>(indices_node, output_shape, axis);
            break;
        case element::Type_t::f64:
            one_hot_v0 = fold_constant_one_hot<double>(indices_node, output_shape, axis);
            break;
        case element::Type_t::i8:
            one_hot_v0 = fold_constant_one_hot<int8_t>(indices_node, output_shape, axis);
            break;
        case element::Type_t::i16:
            one_hot_v0 = fold_constant_one_hot<int16_t>(indices_node, output_shape, axis);
            break;
        case element::Type_t::i32:
            one_hot_v0 = fold_constant_one_hot<int32_t>(indices_node, output_shape, axis);
            break;
        case element::Type_t::i64:
            one_hot_v0 = fold_constant_one_hot<int64_t>(indices_node, output_shape, axis);
            break;
        case element::Type_t::u8:
            one_hot_v0 = fold_constant_one_hot<uint8_t>(indices_node, output_shape, axis);
            break;
        case element::Type_t::u16:
            one_hot_v0 = fold_constant_one_hot<uint16_t>(indices_node, output_shape, axis);
            break;
        case element::Type_t::u32:
            one_hot_v0 = fold_constant_one_hot<uint32_t>(indices_node, output_shape, axis);
            break;
        case element::Type_t::u64:
            one_hot_v0 = fold_constant_one_hot<uint64_t>(indices_node, output_shape, axis);
            break;
        }

        const auto one_hot_data = cast_constant(one_hot_v0, output_type);

        replace_node(m.get_match_root(), one_hot_v0);
        return true;
    };
    auto split_matcher =
        make_shared<pattern::Matcher>(ont_hot_pattern, "ConstantFolding.ConstantOneHot");
    this->add_matcher(split_matcher, one_hot_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
