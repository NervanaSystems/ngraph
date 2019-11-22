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
#include "ngraph/op/concat.hpp"
#include "ngraph/runtime/reference/concat.hpp"
#include "ngraph/type/element_type.hpp"

using namespace std;
using namespace ngraph;

template <typename T>
static shared_ptr<op::Constant> fold_constant_concat_helper(const shared_ptr<op::Concat>& concat)
{
    auto concat_inputs = concat->inputs();
    std::vector<const T*> arg_bufs;
    std::vector<Shape> arg_shapes;

    for (auto& input : concat_inputs)
    {
        auto k = static_cast<op::Constant*>(input.get_source_output().get_node());
        arg_bufs.push_back(k->get_data_ptr<T>());
        arg_shapes.push_back(input.get_shape());
    }

    std::vector<T> result_vec(shape_size(concat->get_shape()));

    runtime::reference::concat<T>(arg_bufs,
                                  result_vec.data(),
                                  arg_shapes,
                                  concat->get_shape(),
                                  concat->get_concatenation_axis());

    return make_shared<op::Constant>(
        concat->get_output_element_type(0), concat->get_output_shape(0), result_vec);
}

void pass::ConstantFolding::construct_constant_concat()
{
    auto concat_op = make_shared<pattern::op::Label>(
        element::f32, Shape{2, 3, 4}, pattern::has_class<op::Concat>());

    auto constant_concat_callback = [](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_concat_callback against node = "
                     << m.get_match_root()->get_name();

        auto concat_node = static_pointer_cast<op::Concat>(m.get_match_root());
        auto concat_inputs = concat_node->inputs();

        if (std::any_of(concat_inputs.begin(), concat_inputs.end(), [](const Input<Node>& input) {
                return !(input.get_source_output().get_node()->is_constant());
            }))
        {
            return false;
        }

        NGRAPH_CHECK(revalidate_and_ensure_static(concat_node));

        std::shared_ptr<op::Constant> replacement;

        switch (concat_node->get_output_element_type(0))
        {
        case element::Type_t::undefined:
            NGRAPH_CHECK(false, "Encountered 'undefined' element type in fold_constant_concat");
            break;
        case element::Type_t::dynamic:
            NGRAPH_CHECK(false, "Encountered 'dynamic' element type in fold_constant_concat");
            break;
        case element::Type_t::u1:
            NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_concat");
            break;
        case element::Type_t::boolean:
            replacement = fold_constant_concat_helper<char>(concat_node);
            break;
        case element::Type_t::bf16:
            replacement = fold_constant_concat_helper<bfloat16>(concat_node);
            break;
        case element::Type_t::f16:
            replacement = fold_constant_concat_helper<float16>(concat_node);
            break;
        case element::Type_t::f32:
            replacement = fold_constant_concat_helper<float>(concat_node);
            break;
        case element::Type_t::f64:
            replacement = fold_constant_concat_helper<double>(concat_node);
            break;
        case element::Type_t::i8:
            replacement = fold_constant_concat_helper<int8_t>(concat_node);
            break;
        case element::Type_t::i16:
            replacement = fold_constant_concat_helper<int16_t>(concat_node);
            break;
        case element::Type_t::i32:
            replacement = fold_constant_concat_helper<int32_t>(concat_node);
            break;
        case element::Type_t::i64:
            replacement = fold_constant_concat_helper<int64_t>(concat_node);
            break;
        case element::Type_t::u8:
            replacement = fold_constant_concat_helper<uint8_t>(concat_node);
            break;
        case element::Type_t::u16:
            replacement = fold_constant_concat_helper<uint16_t>(concat_node);
            break;
        case element::Type_t::u32:
            replacement = fold_constant_concat_helper<uint32_t>(concat_node);
            break;
        case element::Type_t::u64:
            replacement = fold_constant_concat_helper<uint64_t>(concat_node);
            break;
        }

        replace_node(m.get_match_root(), replacement);
        return true;
    };

    auto concat_matcher =
        make_shared<pattern::Matcher>(concat_op, "ConstantFolding.ConstantConcat");
    this->add_matcher(concat_matcher, constant_concat_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
