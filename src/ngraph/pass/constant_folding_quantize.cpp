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
#include "ngraph/op/quantize.hpp"
#include "ngraph/runtime/reference/quantize.hpp"

using namespace std;
using namespace ngraph;

template <class REAL, class QUANT>
Output<Node> fold_constant_quantize(shared_ptr<op::v0::Constant> constant,
                                    shared_ptr<op::v0::Quantize> quant,
                                    shared_ptr<op::v0::Constant> scale,
                                    shared_ptr<op::v0::Constant> offset)
{
    const Shape& out_shape = constant->get_output_shape(0);
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(QUANT));
    QUANT* data_ptr = buffer.get_ptr<QUANT>();

    runtime::reference::quantize<REAL, QUANT>(constant->get_vector<REAL>().data(),
                                              scale->get_vector<REAL>().data(),
                                              offset->get_vector<QUANT>().data(),
                                              data_ptr,
                                              constant->get_output_shape(0),
                                              scale->get_output_shape(0),
                                              quant->get_axes(),
                                              quant->get_round_mode());

    return make_shared<op::v0::Constant>(quant->get_output_element_type(0), out_shape, data_ptr)
        ->output(0);
}

void pass::ConstantFolding::construct_constant_quantize()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::f32, Shape{2}, pattern::has_class<op::v0::Constant>());
    auto q_scale = op::v0::Constant::create(element::f32, Shape{}, {1});
    auto q_offset = op::v0::Constant::create(element::i8, Shape{}, {0});
    auto mode = op::v0::Quantize::RoundMode::ROUND_NEAREST_TOWARD_INFINITY;
    auto quant_op = make_shared<op::v0::Quantize>(
        constant_label, q_scale, q_offset, element::i8, AxisSet{}, mode);
    auto quant = make_shared<pattern::op::Label>(quant_op, nullptr, OutputVector{quant_op});

    auto constant_quantize_callback = [constant_label, quant](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_quantize_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = as_type_ptr<op::v0::Constant>(pattern_map[constant_label]);
        auto quant_match = pattern_map[quant];
        auto quantize_op = as_type_ptr<op::v0::Quantize>(quant_match);

        NGRAPH_CHECK(revalidate_and_ensure_static(quantize_op));

        auto args = quant_match->get_arguments();
        auto scale =
            static_pointer_cast<op::v0::Constant>(quant_match->get_input_node_shared_ptr(1));
        auto offset =
            static_pointer_cast<op::v0::Constant>(quant_match->get_input_node_shared_ptr(2));

        auto type = quant_match->get_output_element_type(0);

        if (constant_match->get_output_element_type(0) != element::f32)
        {
            return false;
        }

        if (type == element::u8)
        {
            m.get_match_value().replace(
                fold_constant_quantize<float, uint8_t>(constant_match, quantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            m.get_match_value().replace(
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
