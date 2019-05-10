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

#include "prior_box.hpp"

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

op::PriorBox::PriorBox(const shared_ptr<Node>& layer_shape,
                       const shared_ptr<Node>& image_shape,
                       const std::vector<float>& min_sizes,
                       const std::vector<float>& max_sizes,
                       const std::vector<float>& aspect_ratios,
                       const bool clip,
                       const bool flip,
                       const float step,
                       const float offset,
                       const std::vector<float>& variances,
                       const bool scale_all)
    : Op("PriorBox", check_single_output_args({layer_shape, image_shape}))
    , m_min_sizes(min_sizes)
    , m_max_sizes(max_sizes)
    , m_aspect_ratios(aspect_ratios)
    , m_clip(clip)
    , m_flip(flip)
    , m_step(step)
    , m_offset(offset)
    , m_variances(variances)
    , m_scale_all(scale_all)
{
    constructor_validate_and_infer_types();
}

void op::PriorBox::validate_and_infer_types()
{
    // shape node should have integer data type. For now we only allow i64
    auto layer_shape_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          layer_shape_et.compatible(element::Type_t::i64),
                          "layer shape input must have element type i64, but has ",
                          layer_shape_et);

    auto image_shape_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          image_shape_et.compatible(element::Type_t::i64),
                          "image shape input must have element type i64, but has ",
                          image_shape_et);

    auto layer_shape_rank = get_input_partial_shape(0).rank();
    auto image_shape_rank = get_input_partial_shape(1).rank();
    NODE_VALIDATION_CHECK(this,
                          layer_shape_rank.compatible(image_shape_rank),
                          "layer shape input rank ",
                          layer_shape_rank,
                          " must match image shape input rank ",
                          image_shape_rank);

    set_input_is_relevant_to_shape(0);

    if (auto const_shape = dynamic_pointer_cast<op::Constant>(get_argument(0)))
    {
        NODE_VALIDATION_CHECK(this,
                              shape_size(const_shape->get_shape()) == 2,
                              "Layer shape must have rank 2",
                              const_shape->get_shape());

        auto layer_shape = const_shape->get_shape_val();
        size_t num_priors = 0;
        // {Prior boxes, Variance-adjusted prior boxes}
        if (m_scale_all)
        {
            num_priors = ((m_flip ? 2 : 1) * m_aspect_ratios.size() + 1) * m_min_sizes.size() +
                         m_max_sizes.size();
        }
        else
        {
            num_priors = (m_flip ? 2 : 1) * m_aspect_ratios.size() + m_min_sizes.size() - 1;
        }

        set_output_type(
            0, element::f32, Shape{2, 4 * layer_shape[0] * layer_shape[1] * num_priors});
    }
    else
    {
        set_output_type(0, element::f32, PartialShape::dynamic());
    }
}

shared_ptr<Node> op::PriorBox::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<PriorBox>(new_args.at(0),
                                 new_args.at(1),
                                 m_min_sizes,
                                 m_max_sizes,
                                 m_aspect_ratios,
                                 m_clip,
                                 m_flip,
                                 m_step,
                                 m_offset,
                                 m_variances,
                                 m_scale_all);
}
