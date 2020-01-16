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

#include "ngraph/op/non_max_suppression.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v1::NonMaxSuppression::type_info;

op::v1::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const Output<Node>& max_output_boxes_per_class,
    const Output<Node>& iou_threshold,
    const Output<Node>& score_threshold,
    const op::v1::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending)
    : Op({boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
{
    constructor_validate_and_infer_types();
}

op::v1::NonMaxSuppression::NonMaxSuppression(
    const Output<Node>& boxes,
    const Output<Node>& scores,
    const op::v1::NonMaxSuppression::BoxEncodingType box_encoding,
    const bool sort_result_descending)
    : Op({boxes,
          scores,
          op::Constant::create(element::i64, Shape{}, {0}),
          op::Constant::create(element::f32, Shape{}, {.0f}),
          op::Constant::create(element::f32, Shape{}, {.0f})})
    , m_box_encoding{box_encoding}
    , m_sort_result_descending{sort_result_descending}
{
    constructor_validate_and_infer_types();
}

shared_ptr<Node> op::v1::NonMaxSuppression::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::NonMaxSuppression>(new_args.at(0),
                                                  new_args.at(1),
                                                  new_args.at(2),
                                                  new_args.at(3),
                                                  new_args.at(4),
                                                  m_box_encoding,
                                                  m_sort_result_descending);
}

void op::v1::NonMaxSuppression::validate_and_infer_types()
{
    const auto boxes_ps = get_input_partial_shape(0);
    const auto scores_ps = get_input_partial_shape(1);
    if (boxes_ps.is_dynamic() || scores_ps.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic(Rank::dynamic()));
        return;
    }

    NODE_VALIDATION_CHECK(this,
                          boxes_ps.rank().is_static() && static_cast<size_t>(boxes_ps.rank()) == 3,
                          "Expected a 3D tensor for the 'boxes' input. Got: ",
                          boxes_ps);

    NODE_VALIDATION_CHECK(this,
                          scores_ps.rank().is_static() &&
                              static_cast<size_t>(scores_ps.rank()) == 3,
                          "Expected a 3D tensor for the 'scores' input. Got: ",
                          scores_ps);

    const auto max_boxes_ps = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this,
                          max_boxes_ps.is_dynamic() || is_scalar(max_boxes_ps.to_shape()),
                          "Expected a scalar for the 'max_output_boxes_per_class' input. Got: ",
                          max_boxes_ps);

    const auto iou_threshold_ps = get_input_partial_shape(3);
    NODE_VALIDATION_CHECK(this,
                          iou_threshold_ps.is_dynamic() || is_scalar(iou_threshold_ps.to_shape()),
                          "Expected a scalar for the 'iou_threshold' input. Got: ",
                          iou_threshold_ps);

    const auto score_threshold_ps = get_input_partial_shape(4);
    NODE_VALIDATION_CHECK(this,
                          score_threshold_ps.is_dynamic() ||
                              is_scalar(score_threshold_ps.to_shape()),
                          "Expected a scalar for the 'score_threshold' input. Got: ",
                          score_threshold_ps);

    const auto num_batches_boxes = boxes_ps[0];
    const auto num_batches_scores = scores_ps[0];
    NODE_VALIDATION_CHECK(this,
                          num_batches_boxes.same_scheme(num_batches_scores),
                          "The first dimension of both 'boxes' and 'scores' must match. Boxes: ",
                          num_batches_boxes,
                          "; Scores: ",
                          num_batches_scores);

    const auto num_boxes_boxes = boxes_ps[1];
    const auto num_boxes_scores = scores_ps[2];
    NODE_VALIDATION_CHECK(this,
                          num_boxes_boxes.same_scheme(num_boxes_scores),
                          "'boxes' and 'scores' input shapes must match at the second and third "
                          "dimension respectively. Boxes: ",
                          num_boxes_boxes,
                          "; Scores: ",
                          num_boxes_scores);

    NODE_VALIDATION_CHECK(this,
                          boxes_ps[2].is_static() && static_cast<size_t>(boxes_ps[2]) == 4u,
                          "The last dimension of the 'boxes' input must be equal to 4. Got:",
                          boxes_ps[2]);

    // NonMaxSuppression produces triplets
    // that have the following format: [batch_index, class_index, box_index]
    PartialShape out_shape = {Dimension::dynamic(), 3};

    const auto max_output_boxes_per_class = input_value(2).get_node_shared_ptr();
    if (num_boxes_boxes.is_static() && scores_ps[1].is_static() &&
        max_output_boxes_per_class->is_constant())
    {
        const auto num_boxes = static_cast<int64_t>(num_boxes_boxes);
        const auto max_output_boxes_per_class = max_boxes_output_from_input();
        const auto num_classes = static_cast<int64_t>(scores_ps[1]);

        out_shape[0] = std::min(num_boxes, max_output_boxes_per_class * num_classes);
    }
    set_output_size(1);
    set_output_type(0, element::i64, out_shape);
}

int64_t op::v1::NonMaxSuppression::max_boxes_output_from_input() const
{
    int64_t max_output_boxes{0};

    const auto max_output_boxes_input =
        as_type_ptr<op::Constant>(input_value(2).get_node_shared_ptr());

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
#endif
    switch (static_cast<element::Type_t>(max_output_boxes_input->get_element_type()))
    {
    case element::Type_t::i8:
    {
        max_output_boxes = max_output_boxes_input->get_vector<int8_t>().at(0);
        break;
    }
    case element::Type_t::i16:
    {
        max_output_boxes = max_output_boxes_input->get_vector<int16_t>().at(0);
        break;
    }
    case element::Type_t::i32:
    {
        max_output_boxes = max_output_boxes_input->get_vector<int32_t>().at(0);
        break;
    }
    case element::Type_t::i64:
    {
        max_output_boxes = max_output_boxes_input->get_vector<int64_t>().at(0);
        break;
    }
    default: break;
    }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

    return max_output_boxes;
}
