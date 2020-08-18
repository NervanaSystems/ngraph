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

#include "ngraph/op/proposal.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Proposal::type_info;

op::v0::Proposal::Proposal(const Output<Node>& class_probs,
                           const Output<Node>& class_logits,
                           const Output<Node>& image_shape,
                           const ProposalAttrs& attrs)
    : Op({class_probs, class_logits, image_shape})
    , m_attrs(attrs)
{
    constructor_validate_and_infer_types();
}

void op::v0::Proposal::validate_and_infer_types()
{
    set_input_is_relevant_to_shape(2);

    const auto& class_probs_pshape = get_input_partial_shape(0);
    const auto& class_logits_pshape = get_input_partial_shape(1);
    const auto& image_shape_pshape = get_input_partial_shape(2);
    if (class_probs_pshape.is_static() && class_logits_pshape.is_static() &&
        image_shape_pshape.is_static())
    {
        const Shape class_probs_shape{class_probs_pshape.to_shape()};
        const Shape class_logits_shape{class_logits_pshape.to_shape()};
        const Shape image_shape_shape{image_shape_pshape.to_shape()};

        NODE_VALIDATION_CHECK(
            this,
            class_probs_shape.size() == 4,
            "Proposal layer shape class_probs input must have rank 4 (class_probs_shape: ",
            class_probs_shape,
            ").");

        NODE_VALIDATION_CHECK(
            this,
            class_logits_shape.size() == 4,
            "Proposal layer shape class_logits_shape input must have rank 4 (class_logits_shape: ",
            class_logits_shape,
            ").");

        NODE_VALIDATION_CHECK(
            this,
            image_shape_shape.size() == 1,
            "Proposal layer image_shape input must have rank 1 (image_shape_shape: ",
            image_shape_shape,
            ").");

        NODE_VALIDATION_CHECK(
            this,
            image_shape_shape[0] >= 3 && image_shape_shape[0] <= 4,
            "Image_shape 1D tensor must have => 3 and <= 4 elements (image_shape_shape[0]",
            image_shape_shape[0],
            ").");

        auto batch_size = class_probs_shape[0];
        set_output_type(0, get_input_element_type(0), Shape{batch_size * m_attrs.post_nms_topn, 5});
    }
    else
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
    }
}

shared_ptr<Node> op::v0::Proposal::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Proposal>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
}

bool op::v0::Proposal::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("attrs.base_size", m_attrs.base_size);
    visitor.on_attribute("attrs.pre_nms_topn", m_attrs.pre_nms_topn);
    visitor.on_attribute("attrs.post_nms_topn", m_attrs.post_nms_topn);
    visitor.on_attribute("attrs.nms_thresh", m_attrs.nms_thresh);
    visitor.on_attribute("attrs.feat_stride", m_attrs.feat_stride);
    visitor.on_attribute("attrs.min_size", m_attrs.min_size);
    visitor.on_attribute("attrs.ratio", m_attrs.ratio);
    visitor.on_attribute("attrs.scale", m_attrs.scale);
    visitor.on_attribute("attrs.clip_before_nms", m_attrs.clip_before_nms);
    visitor.on_attribute("attrs.clip_after_nms", m_attrs.clip_after_nms);
    visitor.on_attribute("attrs.normalize", m_attrs.normalize);
    visitor.on_attribute("attrs.box_size_scale", m_attrs.box_size_scale);
    visitor.on_attribute("attrs.box_coordinate_scale", m_attrs.box_coordinate_scale);
    visitor.on_attribute("attrs.framework", m_attrs.framework);
    return true;
}
