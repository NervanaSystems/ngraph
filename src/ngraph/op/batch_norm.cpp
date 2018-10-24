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

ngraph::op::BatchNormBase::BatchNormBase(const std::string& node_type,
                                         double eps,
                                         const NodeVector& args)
    : Op(node_type, check_single_output_args(args))
    , m_epsilon(eps)
{
    constructor_validate_and_infer_types();
}

ngraph::op::BatchNormInference::BatchNormInference(double eps,
                                                   std::shared_ptr<ngraph::Node> gamma,
                                                   std::shared_ptr<ngraph::Node> beta,
                                                   std::shared_ptr<ngraph::Node> input,
                                                   std::shared_ptr<ngraph::Node> mean,
                                                   std::shared_ptr<ngraph::Node> variance)
    : BatchNormBase(
          "BatchNormInference", eps, check_single_output_args({gamma, beta, input, mean, variance}))
{
    constructor_validate_and_infer_types();
}

ngraph::op::BatchNormTraining::BatchNormTraining(double eps,
                                                 std::shared_ptr<ngraph::Node> gamma,
                                                 std::shared_ptr<ngraph::Node> beta,
                                                 std::shared_ptr<ngraph::Node> input)
    : BatchNormBase("BatchNormTraining", eps, check_single_output_args({gamma, beta, input}))
{
    constructor_validate_and_infer_types();
}

ngraph::op::BatchNormTraining::BatchNormTraining(double eps,
                                                 std::shared_ptr<ngraph::Node> gamma,
                                                 std::shared_ptr<ngraph::Node> beta,
                                                 std::shared_ptr<ngraph::Node> input,
                                                 std::shared_ptr<ngraph::Node> mean,
                                                 std::shared_ptr<ngraph::Node> variance)
    : BatchNormBase(
          "BatchNormTraining", eps, check_single_output_args({gamma, beta, input, mean, variance}))
{
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNormInference::validate_and_infer_types()
{
    if (validate_punt_if_dynamic())
    {
        return;
    }

    auto bn_input_shape = get_input_shape(INPUT);
    BatchNormBase::validate_and_infer_types();
    auto& et = get_input_element_type(INPUT);
    set_output_size(1);
    set_output_type(0, et, bn_input_shape);
}

void ngraph::op::BatchNormTraining::validate_and_infer_types()
{
    if (validate_punt_if_dynamic())
    {
        return;
    }

    auto bn_input_shape = get_input_shape(INPUT);
    BatchNormBase::validate_and_infer_types();
    auto& et = get_input_element_type(INPUT);
    Shape channel_shape{bn_input_shape[1]};
    set_output_size(3);
    set_output_type(0, et, bn_input_shape);
    set_output_type(1, et, channel_shape);
    set_output_type(2, et, channel_shape);
}

void ngraph::op::BatchNormBase::validate_and_infer_types()
{
    auto bn_input_shape = get_input_shape(INPUT);
    NODE_VALIDATION_ASSERT(this, bn_input_shape.size() >= 2)
        << "Input argument must have rank of at least 2 (input argument shape: " << bn_input_shape
        << ").";

    NODE_VALIDATION_ASSERT(this, bn_input_shape[1] != 0)
        << "Input argument's channel dimension must have size of at least 1 (input argument shape: "
        << bn_input_shape << ").";

    auto& et = get_input_element_type(INPUT);

    Shape channel_shape{bn_input_shape[1]};

    const char* input_names[]{"gamma", "beta", "input", "mean", "variance"};

    for (size_t i = 0; i < get_input_size(); i++)
    {
        if (i == INPUT)
        {
            continue;
        }

        NODE_VALIDATION_ASSERT(this, get_input_element_type(i) == et)
            << "Element type of " << input_names[i] << " (" << get_input_element_type(i)
            << ") is not equal to the element type of input (" << et << ").";

        NODE_VALIDATION_ASSERT(this, get_input_shape(i) == channel_shape)
            << "Shape of " << input_names[i] << " must match the channel dimension of the "
            << "input data (expected shape: " << channel_shape << ", actual shape of "
            << input_names[i] << ": " << get_input_shape(i)
            << ", shape of input: " << bn_input_shape << ").";
    }
}

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormInference::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormInference>(
        m_epsilon, new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormTraining::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<BatchNormTraining>(
        m_epsilon, new_args.at(0), new_args.at(1), new_args.at(2));
}

ngraph::op::BatchNormTrainingBackprop::BatchNormTrainingBackprop(
    double eps,
    std::shared_ptr<ngraph::Node> gamma,
    std::shared_ptr<ngraph::Node> beta,
    std::shared_ptr<ngraph::Node> input,
    std::shared_ptr<ngraph::Node> mean,
    std::shared_ptr<ngraph::Node> variance,
    std::shared_ptr<ngraph::Node> delta)
    : Op("BatchNormTrainingBackprop",
         check_single_output_args({gamma, beta, input, mean, variance, delta}))
    , epsilon(eps)

{
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNormTrainingBackprop::validate_and_infer_types()
{
    if (validate_punt_if_dynamic())
    {
        return;
    }

    set_output_size(3);

    NODE_VALIDATION_ASSERT(this, get_input_shape(INPUT).size() == 4)
        << "Input data shape is not a 4D tensor (input data shape: " << get_input_shape(INPUT)
        << ").";

    auto et = get_input_element_type(INPUT);
    const char* input_names[] = {"gamma", "beta", "input", "mean", "variance", "delta"};

    Shape channel_shape{get_input_shape(INPUT)[1]};

    for (size_t i = 0; i < get_input_size(); i++)
    {
        NODE_VALIDATION_ASSERT(this, get_input_element_type(i) == et)
            << "Element type of " << input_names[i] << " (" << get_input_element_type(i)
            << ") is not equal to the element type of input (" << et << ").";

        // Note that the shape of delta, a special case, will be checked after the loop.
        if (i == DELTA || i == INPUT)
        {
            continue;
        }

        NODE_VALIDATION_ASSERT(this, get_input_shape(i) == channel_shape)
            << "Shape of " << input_names[i] << " must match the channel dimension of the "
            << "input data (expected shape: " << channel_shape << ", actual shape of "
            << input_names[i] << ": " << get_input_shape(i)
            << ", shape of input: " << get_input_shape(INPUT) << ").";
    }

    NODE_VALIDATION_ASSERT(this, get_input_shape(DELTA) == get_input_shape(INPUT))
        << "Shape of delta must match the shape of the input data (expected shape: "
        << get_input_shape(INPUT) << ", actual shape of delta: " << get_input_shape(DELTA) << ").";

    set_output_type(0, get_input_element_type(INPUT), get_input_shape(INPUT));
    set_output_type(1, get_input_element_type(GAMMA), get_input_shape(GAMMA));
    set_output_type(2, get_input_element_type(BETA), get_input_shape(BETA));
}

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormTrainingBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<op::BatchNormTrainingBackprop>(epsilon,
                                                           new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4),
                                                           new_args.at(5));
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
        get_eps_value(), gamma, beta, input, mean, var, deltas.at(0));
    auto dinput = std::make_shared<op::GetOutputElement>(bbn, 0);
    auto dgamma = std::make_shared<op::GetOutputElement>(bbn, 1);
    auto dbeta = std::make_shared<op::GetOutputElement>(bbn, 2);

    adjoints.add_delta(input, dinput);
    adjoints.add_delta(gamma, dgamma);
    adjoints.add_delta(beta, dbeta);
}
