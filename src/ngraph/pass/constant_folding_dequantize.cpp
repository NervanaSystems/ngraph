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
#include "ngraph/op/dequantize.hpp"
#include "ngraph/runtime/reference/dequantize.hpp"

using namespace std;
using namespace ngraph;

template <class QUANT, class REAL>
Output<Node> fold_constant_dequantize(shared_ptr<op::v0::Constant> constant,
                                      shared_ptr<op::v0::Dequantize> dequant,
                                      shared_ptr<op::v0::Constant> scale,
                                      shared_ptr<op::v0::Constant> offset)
{
    const Shape& out_shape = constant->get_output_shape(0);
    runtime::AlignedBuffer buffer(shape_size(out_shape) * sizeof(REAL));
    REAL* data_ptr = buffer.get_ptr<REAL>();

    runtime::reference::dequantize<QUANT, REAL>(constant->get_vector<QUANT>().data(),
                                                scale->get_vector<REAL>().data(),
                                                offset->get_vector<QUANT>().data(),
                                                data_ptr,
                                                constant->get_output_shape(0),
                                                scale->get_output_shape(0),
                                                dequant->get_axes());

    return make_shared<op::v0::Constant>(dequant->get_output_element_type(0), out_shape, data_ptr)
        ->output(0);
}

void pass::ConstantFolding::construct_constant_dequantize()
{
    auto constant_label = make_shared<pattern::op::Label>(
        element::u8, Shape{2}, pattern::has_class<op::v0::Constant>());
    auto dq_scale = op::v0::Constant::create(element::f32, Shape{}, {1});
    auto dq_offset = op::v0::Constant::create(element::u8, Shape{}, {1});
    auto dequant_op = make_shared<op::v0::Dequantize>(
        constant_label, dq_scale, dq_offset, element::f32, AxisSet{});
    auto dequant = make_shared<pattern::op::Label>(dequant_op, nullptr, OutputVector{dequant_op});

    auto constant_dequantize_callback = [constant_label, dequant](pattern::Matcher& m) {
        NGRAPH_DEBUG << "In callback for constant_dequantize_callback against node = "
                     << m.get_match_root()->get_name();

        auto pattern_map = m.get_pattern_map();

        auto constant_match = as_type_ptr<op::v0::Constant>(pattern_map[constant_label]);
        auto dequant_match = pattern_map[dequant];
        auto dequantize_op = as_type_ptr<op::v0::Dequantize>(dequant_match);

        auto scale =
            as_type_ptr<op::v0::Constant>(dequant_match->input_value(1).get_node_shared_ptr());
        auto offset =
            as_type_ptr<op::v0::Constant>(dequant_match->input_value(2).get_node_shared_ptr());

        NGRAPH_CHECK(revalidate_and_ensure_static(dequantize_op));
        auto type = constant_match->get_output_element_type(0);

        if (dequant_match->get_output_element_type(0) != element::f32)
        {
            return false;
        }

        if (type == element::u8)
        {
            m.get_match_value().replace(fold_constant_dequantize<uint8_t, float>(
                constant_match, dequantize_op, scale, offset));
            return true;
        }
        else if (type == element::i8)
        {
            m.get_match_value().replace(fold_constant_dequantize<int8_t, float>(
                constant_match, dequantize_op, scale, offset));
            return true;
        }

        return false;
    };

    auto dequantize_matcher =
        make_shared<pattern::Matcher>(dequant, "ConstantFolding.ConstantDequantize");
    this->add_matcher(
        dequantize_matcher, constant_dequantize_callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
