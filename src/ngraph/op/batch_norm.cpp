//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

ngraph::op::BatchNormTraining::BatchNormTraining(std::shared_ptr<ngraph::Node> input,
                                                 std::shared_ptr<ngraph::Node> gamma,
                                                 std::shared_ptr<ngraph::Node> beta,
                                                 double epsilon)
    : Op("BatchNormTraining", check_single_output_args({gamma, beta, input}))
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

// DEPRECATED
ngraph::op::BatchNormTraining::BatchNormTraining(double eps,
                                                 std::shared_ptr<ngraph::Node> gamma,
                                                 std::shared_ptr<ngraph::Node> beta,
                                                 std::shared_ptr<ngraph::Node> input)
    : Op("BatchNormTraining", check_single_output_args({gamma, beta, input}))
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

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormTraining::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormTraining>(
        new_args.at(2), new_args.at(0), new_args.at(1), m_epsilon);
}

void ngraph::op::BatchNormTraining::generate_adjoints(autodiff::Adjoints& adjoints,
                                                      const NodeVector& deltas)
{
    auto gamma = get_argument(0);
    auto beta = get_argument(1);
    auto input = get_argument(2);
    std::shared_ptr<Node> mean = nullptr;
    std::shared_ptr<Node> var = nullptr;

    // Extract mean and variance outputs from BatchNormBase
    // as these are used by BatchNormTrainingBackprop.
    // The users of the outputs (GetOutputElements' Inputs) aren't sorted
    // and get_n() is used to sort the inputs in the same order as Batchnorm's outputs
    // Next, Mean and Variance (`at(1)` and `at(2)`) are extracted
    // Please see `add_output` in `BatchNormBase::BatchNormBase` for more details

    auto goes = op::get_output_elements(shared_from_this());
    mean = goes.at(1);
    var = goes.at(2);
    if (!mean)
    {
        throw ngraph_error("GetOutputElement for mean is missing");
    }

    if (!var)
    {
        throw ngraph_error("GetOutputElement for variance is missing");
    }

    auto bbn = std::make_shared<op::BatchNormTrainingBackprop>(
        input, gamma, beta, mean, var, deltas.at(0), get_eps_value());
    auto dinput = std::make_shared<op::GetOutputElement>(bbn, 0);
    auto dgamma = std::make_shared<op::GetOutputElement>(bbn, 1);
    auto dbeta = std::make_shared<op::GetOutputElement>(bbn, 2);

    adjoints.add_delta(input, dinput);
    adjoints.add_delta(gamma, dgamma);
    adjoints.add_delta(beta, dbeta);
}

ngraph::op::BatchNormInference::BatchNormInference(std::shared_ptr<ngraph::Node> input,
                                                   std::shared_ptr<ngraph::Node> gamma,
                                                   std::shared_ptr<ngraph::Node> beta,
                                                   std::shared_ptr<ngraph::Node> mean,
                                                   std::shared_ptr<ngraph::Node> variance,
                                                   double epsilon)
    : Op("BatchNormInference", check_single_output_args({gamma, beta, input, mean, variance}))
    , m_epsilon(epsilon)
{
    constructor_validate_and_infer_types();
}

// DEPRECATED
ngraph::op::BatchNormInference::BatchNormInference(double eps,
                                                   std::shared_ptr<ngraph::Node> gamma,
                                                   std::shared_ptr<ngraph::Node> beta,
                                                   std::shared_ptr<ngraph::Node> input,
                                                   std::shared_ptr<ngraph::Node> mean,
                                                   std::shared_ptr<ngraph::Node> variance)
    : Op("BatchNormInference", check_single_output_args({gamma, beta, input, mean, variance}))
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

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormInference::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(
        new_args.at(2), new_args.at(0), new_args.at(1), new_args.at(3), new_args.at(4), m_epsilon);
}

ngraph::op::BatchNormTrainingBackprop::BatchNormTrainingBackprop(
    std::shared_ptr<ngraph::Node> input,
    std::shared_ptr<ngraph::Node> gamma,
    std::shared_ptr<ngraph::Node> beta,
    std::shared_ptr<ngraph::Node> mean,
    std::shared_ptr<ngraph::Node> variance,
    std::shared_ptr<ngraph::Node> delta,
    double epsilon)
    : Op("BatchNormTrainingBackprop",
         check_single_output_args({gamma, beta, input, mean, variance, delta}))
    , m_epsilon(epsilon)

{
    set_output_size(3);
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNormTrainingBackprop::validate_and_infer_types()
{
    PartialShape input_and_delta_shape{get_input_partial_shape(INPUT_DATA)};

    NODE_VALIDATION_ASSERT(
        this, PartialShape::merge_into(input_and_delta_shape, get_input_partial_shape(INPUT_DELTA)))
        << "Shape of delta does not match the shape of the input data (input data shape: "
        << get_input_partial_shape(INPUT_DATA)
        << ", delta shape: " << get_input_partial_shape(INPUT_DELTA) << ").";

    element::Type input_and_delta_et;

    NODE_VALIDATION_ASSERT(this,
                           element::Type::merge(input_and_delta_et,
                                                get_input_element_type(INPUT_DATA),
                                                get_input_element_type(INPUT_DELTA)))
        << "Element type for input (" << get_input_element_type(INPUT_DATA)
        << ") does not match element type for delta (" << get_input_element_type(INPUT_DATA)
        << ").";

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

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormTrainingBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<op::BatchNormTrainingBackprop>(new_args.at(2),
                                                           new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           new_args.at(5),
                                                           m_epsilon);
}
