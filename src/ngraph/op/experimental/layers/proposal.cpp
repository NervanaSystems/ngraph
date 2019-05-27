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

#include "proposal.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::Proposal::Proposal(const std::shared_ptr<Node>& class_probs,
                       const std::shared_ptr<Node>& class_logits,
                       const std::shared_ptr<Node>& image_shape,
                       const ProposalAttrs& attrs)
    : Op("Proposal", check_single_output_args({class_probs, class_logits, image_shape}))
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::Proposal::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    auto image_shape_et = get_input_element_type(2);
    NODE_VALIDATION_CHECK(this,
                          image_shape_et.compatible(element::Type_t::i64),
                          "image shape input must have element type i64, but has ",
                          image_shape_et);

    set_input_is_relevant_to_shape(2);

    if (auto const_shape = dynamic_pointer_cast<op::Constant>(get_argument(2)))
    {
        NODE_VALIDATION_CHECK(this,
                              shape_size(const_shape->get_shape()) == 2,
                              "Layer shape must have rank 2",
                              const_shape->get_shape());

        auto image_shape = const_shape->get_shape_val();

        set_output_type(0, element::f32, Shape{image_shape[0] * m_attrs.post_nms_topn, 5});
    }
    else
    {
        set_output_type(0, element::f32, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::Proposal::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}
