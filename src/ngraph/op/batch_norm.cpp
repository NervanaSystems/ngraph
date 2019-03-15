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

#include <set>
#include <sstream>

#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/validation_util.hpp"

ngraph::op::BatchNormTraining::BatchNormTraining(const NodeOutput& input,
                                                 const NodeOutput& gamma,
                                                 const NodeOutput& beta,
                                                 double epsilon)
    : Op("BatchNormTraining", {gamma, beta, input})
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

// DEPRECATED
ngraph::op::BatchNormTraining::BatchNormTraining(double eps,
                                                 const NodeOutput& gamma,
                                                 const NodeOutput& beta,
                                                 const NodeOutput& input)
    : Op("BatchNormTraining", {gamma, beta, input})
    , m_epsilon(eps)
{
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNormTraining::validate_and_infer_types()
{
    element::Type result_et;
    PartialShape result_batch_shape;
    PartialShape result_channel_shape;

    set_output_size(3);
    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
                                 get_input_element_type(INPUT_DATA),
                                 get_input_element_type(INPUT_GAMMA),
                                 get_input_element_type(INPUT_BETA),
                                 get_input_partial_shape(INPUT_DATA),
                                 get_input_partial_shape(INPUT_GAMMA),
                                 get_input_partial_shape(INPUT_BETA));

    set_output_type(0, result_et, result_batch_shape);
    set_output_type(1, result_et, result_channel_shape);
    set_output_type(2, result_et, result_channel_shape);
}

std::shared_ptr<ngraph::Node> ngraph::op::BatchNormTraining::copy_with_new_source_outputs(
    const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return std::make_shared<BatchNormTraining>(
        new_source_outputs.at(2), new_source_outputs.at(0), new_source_outputs.at(1), m_epsilon);
}

void ngraph::op::BatchNormTraining::build_backprop(autodiff::Adjoints& adjoints,
                                                   const OutputVector& deltas)
{
    auto gamma = get_input_source_output(0);
    auto beta = get_input_source_output(1);
    auto input = get_input_source_output(2);

    auto mean = NodeOutput(shared_from_this(), 1);
    auto var = NodeOutput(shared_from_this(), 2);

    auto bbn = std::make_shared<op::BatchNormTrainingBackprop>(
        input, gamma, beta, mean, var, deltas.at(0), get_eps_value());

    auto dinput = NodeOutput(bbn, 0);
    auto dgamma = NodeOutput(bbn, 1);
    auto dbeta = NodeOutput(bbn, 2);

    adjoints.add_output_delta(input, dinput);
    adjoints.add_output_delta(gamma, dgamma);
    adjoints.add_output_delta(beta, dbeta);
}

ngraph::op::BatchNormInference::BatchNormInference(const NodeOutput& input,
                                                   const NodeOutput& gamma,
                                                   const NodeOutput& beta,
                                                   const NodeOutput& mean,
                                                   const NodeOutput& variance,
                                                   double epsilon)
    : Op("BatchNormInference", {gamma, beta, input, mean, variance})
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

// DEPRECATED
ngraph::op::BatchNormInference::BatchNormInference(double eps,
                                                   const NodeOutput& gamma,
                                                   const NodeOutput& beta,
                                                   const NodeOutput& input,
                                                   const NodeOutput& mean,
                                                   const NodeOutput& variance)
    : Op("BatchNormInference", {gamma, beta, input, mean, variance})
    , m_epsilon(eps)
{
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNormInference::validate_and_infer_types()
{
    element::Type result_et;
    PartialShape result_batch_shape;
    PartialShape result_channel_shape; // unused here

    set_output_size(1);
    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
                                 get_input_element_type(INPUT_DATA),
                                 get_input_element_type(INPUT_GAMMA),
                                 get_input_element_type(INPUT_BETA),
                                 get_input_element_type(INPUT_MEAN),
                                 get_input_element_type(INPUT_VARIANCE),
                                 get_input_partial_shape(INPUT_DATA),
                                 get_input_partial_shape(INPUT_GAMMA),
                                 get_input_partial_shape(INPUT_BETA),
                                 get_input_partial_shape(INPUT_MEAN),
                                 get_input_partial_shape(INPUT_VARIANCE));

    set_output_type(0, result_et, result_batch_shape);
}

std::shared_ptr<ngraph::Node> ngraph::op::BatchNormInference::copy_with_new_source_outputs(
    const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return std::make_shared<BatchNormInference>(new_source_outputs.at(2),
                                                new_source_outputs.at(0),
                                                new_source_outputs.at(1),
                                                new_source_outputs.at(3),
                                                new_source_outputs.at(4),
                                                m_epsilon);
}

ngraph::op::BatchNormTrainingBackprop::BatchNormTrainingBackprop(const NodeOutput& input,
                                                                 const NodeOutput& gamma,
                                                                 const NodeOutput& beta,
                                                                 const NodeOutput& mean,
                                                                 const NodeOutput& variance,
                                                                 const NodeOutput& delta,
                                                                 double epsilon)
    : Op("BatchNormTrainingBackprop", {gamma, beta, input, mean, variance, delta})
    , m_epsilon(epsilon)

{
    set_output_size(3);
    constructor_validate_and_infer_types();
}

ngraph::op::BatchNormTrainingBackprop::BatchNormTrainingBackprop(double epsilon,
                                                                 const NodeOutput& gamma,
                                                                 const NodeOutput& beta,
                                                                 const NodeOutput& input,
                                                                 const NodeOutput& mean,
                                                                 const NodeOutput& variance,
                                                                 const NodeOutput& delta)
    : Op("BatchNormTrainingBackprop", {gamma, beta, input, mean, variance, delta})
    , m_epsilon(epsilon)

{
    set_output_size(3);
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNormTrainingBackprop::validate_and_infer_types()
{
    PartialShape input_and_delta_shape{get_input_partial_shape(INPUT_DATA)};

    NODE_VALIDATION_CHECK(
        this,
        PartialShape::merge_into(input_and_delta_shape, get_input_partial_shape(INPUT_DELTA)),
        "Shape of delta does not match the shape of the input data (input data shape: ",
        get_input_partial_shape(INPUT_DATA),
        ", delta shape: ",
        get_input_partial_shape(INPUT_DELTA),
        ").");

    element::Type input_and_delta_et;

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(input_and_delta_et,
                                               get_input_element_type(INPUT_DATA),
                                               get_input_element_type(INPUT_DELTA)),
                          "Element type for input (",
                          get_input_element_type(INPUT_DATA),
                          ") does not match element type for delta (",
                          get_input_element_type(INPUT_DATA),
                          ").");

    element::Type result_et;
    PartialShape result_batch_shape;
    PartialShape result_channel_shape;

    std::tie(result_et, result_batch_shape, result_channel_shape) =
        infer_batch_norm_forward(this,
                                 input_and_delta_et,
                                 get_input_element_type(INPUT_GAMMA),
                                 get_input_element_type(INPUT_BETA),
                                 get_input_element_type(INPUT_MEAN),
                                 get_input_element_type(INPUT_VARIANCE),
                                 input_and_delta_shape,
                                 get_input_partial_shape(INPUT_GAMMA),
                                 get_input_partial_shape(INPUT_BETA),
                                 get_input_partial_shape(INPUT_MEAN),
                                 get_input_partial_shape(INPUT_VARIANCE));

    set_output_type(0, result_et, result_batch_shape);
    set_output_type(1, result_et, result_channel_shape);
    set_output_type(2, result_et, result_channel_shape);
}

std::shared_ptr<ngraph::Node> ngraph::op::BatchNormTrainingBackprop::copy_with_new_source_outputs(
    const OutputVector& new_source_outputs) const
{
    check_new_source_outputs_count(this, new_source_outputs);
    return std::make_shared<op::BatchNormTrainingBackprop>(new_source_outputs.at(2),
                                                           new_source_outputs.at(0),
                                                           new_source_outputs.at(1),
                                                           new_source_outputs.at(3),
                                                           new_source_outputs.at(4),
                                                           new_source_outputs.at(5),
                                                           m_epsilon);
}
