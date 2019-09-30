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
#include "ngraph/op/quantize.hpp"
#include "ngraph/runtime/reference/quantize.hpp"

using namespace std;
using namespace ngraph;

template <class REAL, class QUANT>
shared_ptr<op::Constant> fold_constant_quantize(shared_ptr<op::Constant> constant,
                                                shared_ptr<op::Quantize> quant,
                                                shared_ptr<op::Constant> scale,
                                                shared_ptr<op::Constant> offset)
{
    auto out_shape = constant->get_shape();
    vector<QUANT> out_vec(shape_size(out_shape));

    runtime::reference::quantize<REAL, QUANT>(constant->get_vector<REAL>().data(),
                                              scale->get_vector<REAL>().data(),
                                              offset->get_vector<QUANT>().data(),
                                              out_vec.data(),
                                              constant->get_shape(),
                                              scale->get_shape(),
                                              quant->get_axes(),
                                              quant->get_round_mode());

    return make_shared<op::Constant>(quant->get_element_type(), out_shape, out_vec);
}

void pass::ConstantFolding::construct_constant_quantize()
{
    auto constant_label =
        make_shared<pattern::op::Label>(element::f32, Shape{2}, pattern::has_class<op::Constant>());
    auto q_scale = op::Constant::create(element::f32, Shape{}, {1});
    auto q_offset = op::Constant::create(element::i8, Shape{}, {0});
    auto mode = op::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;
    auto quant_op =
        make_shared<op::Quantize>(constant_label, q_scale, q_offset, element::i8, AxisSet{}, mode);
    auto quant = make_shared<pattern::op::Label>(quant_op, nullptr, NodeVector{quant_op});

    auto constant_quantize_callback = [constant_label, quant](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_quantize_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = as_type_ptr<op::Constant>(pattern_map[constant_label]);
        auto quant_match = pattern_map[quant];
        auto quantize_op = as_type_ptr<op::Quantize>(quant_match);

        NGRAPH_CHECK(revalidate_and_ensure_static(quantize_op));

        auto args = quant_match->get_arguments();
        auto scale = static_pointer_cast<op::Constant>(
            quant_match->input(1).get_source_output().get_node_shared_ptr());
        auto offset = static_pointer_cast<op::Constant>(
            quant_match->input(2).get_source_output().get_node_shared_ptr());

        auto type = quant_match->get_element_type();

        if (constant_match->get_element_type() != element::f32)
        {
            return false;
        }

        if (type == element::u8)
        {
            replace_node(
                m.get_match_root(),
                fold_constant_quantize<float, uint8_t>(constant_match, quantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            replace_node(
                m.get_match_root(),
                fold_constant_quantize<float, int8_t>(constant_match, quantize_op, scale, offset));
            return true;
        }

        return false;
    };

    auto quantize_matcher =
        make_shared<pattern::Matcher>(quant, "ConstantFolding.ConstantQuantize");
    this->add_matcher(
        quantize_matcher, constant_quantize_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
