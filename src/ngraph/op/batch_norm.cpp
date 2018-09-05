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

ngraph::op::BatchNorm::BatchNorm(double eps,
                                 std::shared_ptr<ngraph::Node> gamma,
                                 std::shared_ptr<ngraph::Node> beta,
                                 std::shared_ptr<ngraph::Node> input)
    : Op("BatchNorm", check_single_output_args({gamma, beta, input}))
    , m_bn_input_shape(input->get_shape())
    , m_epsilon(eps)
    , m_training(true)
{
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNorm::validate_and_infer_types()
{
    m_bn_input_shape = get_input_shape(INPUT);
    NODE_VALIDATION_ASSERT(this, m_bn_input_shape.size() >= 2)
        << "Input argument must have rank of at least 2 (input argument shape: " << m_bn_input_shape
        << ").";

    NODE_VALIDATION_ASSERT(this, m_bn_input_shape[1] != 0)
        << "Input argument's channel dimension must have size of at least 1 (input argument shape: "
        << m_bn_input_shape << ").";

    auto& et = get_input_element_type(INPUT);
    auto in_size = get_input_size();

    NODE_VALIDATION_ASSERT(this, in_size == 3 || in_size == 5)
        << "Argument count must be either 3 or 5 (received argument count: " << in_size << ").";

    Shape channel_shape{m_bn_input_shape[1]};

    if (in_size == 3)
    {
        set_output_size(3);
        m_bn_mean_shape = channel_shape;
        set_output_type(1, et, m_bn_mean_shape);
        m_bn_variance_shape = channel_shape;
        set_output_type(2, et, m_bn_variance_shape);
    }
    else
    {
        set_output_size(1);
    }

    set_output_type(0, et, m_bn_input_shape);

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
            << ", shape of input: " << m_bn_input_shape << ").";
    }
}

ngraph::op::BatchNorm::BatchNorm(double eps,
                                 std::shared_ptr<ngraph::Node> gamma,
                                 std::shared_ptr<ngraph::Node> beta,
                                 std::shared_ptr<ngraph::Node> input,
                                 std::shared_ptr<ngraph::Node> mean,
                                 std::shared_ptr<ngraph::Node> variance,
                                 bool training)
    : Op("BatchNorm", check_single_output_args({gamma, beta, input, mean, variance}))
    , m_epsilon(eps)
    , m_training(training)
{
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNorm::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);

    if (m_training)
    {
        // FIXME(amprocte): is this redundant?
        NODE_VALIDATION_ASSERT(this, new_args.size() == 3 || new_args.size() == 5);

        if (new_args.size() == 3)
        {
            return std::make_shared<BatchNorm>(
                m_epsilon, new_args.at(0), new_args.at(1), new_args.at(2));
        }
        else
        {
            return std::make_shared<BatchNorm>(m_epsilon,
                                               new_args.at(0),
                                               new_args.at(1),
                                               new_args.at(2),
                                               new_args.at(3),
                                               new_args.at(4),
                                               true);
        }
    }
    else
    {
        NODE_VALIDATION_ASSERT(this, new_args.size() == 5);

        return std::make_shared<BatchNorm>(m_epsilon,
                                           new_args.at(0),
                                           new_args.at(1),
                                           new_args.at(2),
                                           new_args.at(3),
                                           new_args.at(4),
                                           false);
    }
}

ngraph::op::BatchNormBackprop::BatchNormBackprop(double eps,
                                                 std::shared_ptr<ngraph::Node> gamma,
                                                 std::shared_ptr<ngraph::Node> beta,
                                                 std::shared_ptr<ngraph::Node> input,
                                                 std::shared_ptr<ngraph::Node> mean,
                                                 std::shared_ptr<ngraph::Node> variance,
                                                 std::shared_ptr<ngraph::Node> delta)
    : Op("BatchNormBackprop", check_single_output_args({gamma, beta, input, mean, variance, delta}))
    , epsilon(eps)

{
    constructor_validate_and_infer_types();
}

void ngraph::op::BatchNormBackprop::validate_and_infer_types()
{
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
    ngraph::op::BatchNormBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<op::BatchNormBackprop>(epsilon,
                                                   new_args.at(0),
                                                   new_args.at(1),
                                                   new_args.at(2),
                                                   new_args.at(3),
                                                   new_args.at(4),
                                                   new_args.at(5));
}

void ngraph::op::BatchNorm::generate_adjoints(autodiff::Adjoints& adjoints,
                                              const NodeVector& deltas)
{
    auto gamma = get_argument(0);
    auto beta = get_argument(1);
    auto input = get_argument(2);
    std::shared_ptr<Node> mean = nullptr;
    std::shared_ptr<Node> var = nullptr;

    if (!this->get_training_flag())
    {
        throw ngraph_error("generate_adjoints called on BatchNormInference op " + this->get_name());
    }
    // Extract mean and variance outputs from BatchNorm
    // as these are used by BatchNormBackprop.
    // The users of the outputs (GetOutputElements' Inputs) aren't sorted
    // and get_n() is used to sort the inputs in the same order as Batchnorm's outputs
    // Next, Mean and Variance (`at(1)` and `at(2)`) are extracted
    // Please see `add_output` in `BatchNorm::BatchNorm` for more details
    if (this->get_training_flag() && get_input_size() == 3)
    {
        auto goes = op::get_output_elements(shared_from_this());
        mean = goes.at(1);
        var = goes.at(2);
        if (!mean)
        {
            throw ngraph_error("GetOutputElement for mean is missing");
        };
        if (!var)
        {
            throw ngraph_error("GetOutputElement for variance is missing");
        }
    }
    else // BatchNorm Training with global stats
    {
        mean = get_argument(3);
        var = get_argument(4);
    }
    auto bbn = std::make_shared<op::BatchNormBackprop>(
        get_eps_value(), gamma, beta, input, mean, var, deltas.at(0));
    auto dinput = std::make_shared<op::GetOutputElement>(bbn, 0);
    auto dgamma = std::make_shared<op::GetOutputElement>(bbn, 1);
    auto dbeta = std::make_shared<op::GetOutputElement>(bbn, 2);

    adjoints.add_delta(input, dinput);
    adjoints.add_delta(gamma, dgamma);
    adjoints.add_delta(beta, dbeta);
}
