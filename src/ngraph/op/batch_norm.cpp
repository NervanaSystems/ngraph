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
    if (m_bn_input_shape.size() < 2)
    {
        throw ngraph_error("input tensor to batchnorm must have tensor of at least rank 2");
    }
    if (m_bn_input_shape[1] == 0)
    {
        throw ngraph_error("input tensor must have at least one channel for batch normalization");
    }

    auto& et = get_input_element_type(INPUT);
    auto in_size = get_input_size();
    if (in_size == 3)
    {
        set_output_size(3);
        this->m_bn_mean_shape.push_back(m_bn_input_shape[1]);
        set_output_type(1, et, m_bn_mean_shape);
        this->m_bn_variance_shape.push_back(m_bn_input_shape[1]);
        set_output_type(2, et, m_bn_variance_shape);
    }
    else if (in_size == 5)
    {
        set_output_size(1);
    }
    else
    {
        throw ngraph_error("Invalid BatchNorm args");
    }

    set_output_type(0, et, m_bn_input_shape);

    Shape channel_shape{m_bn_input_shape[1]};
    const char* input_names[]{"gamma", "beta", "input", "mean", "variance"};

    for (size_t i = 0; i < get_input_size(); i++)
    {
        if (i == 2)
        {
            continue;
        }
        if (get_input_element_type(i) != et)
        {
            std::stringstream err_msg;
            err_msg << "The element type " << get_input_element_type(i) << " of input "
                    << input_names[i] << " isn't equal to the input data's type " << et;
            throw ngraph_error(err_msg.str());
        }

        if (get_input_shape(i) != channel_shape)
        {
            std::stringstream err_msg;
            err_msg << "The shape " << get_input_shape(i) << " of " << input_names[i]
                    << " isn't equal to input channel's shape " << channel_shape;
            throw ngraph_error(err_msg.str());
        }
    }

    for (size_t index = 0; index < get_input_size(); index++)
    {
        if (index != INPUT && get_input_shape(index).size() != 1)
        {
            auto err_msg = std::string(input_names[index]) + " should have rank of 1";
            throw ngraph_error(err_msg.c_str());
        }

        if (index != INPUT && get_input_shape(index)[0] != m_bn_input_shape[1])
        {
            auto err_msg = std::string(input_names[index]) +
                           " shape should match the input channel size (" +
                           std::to_string(m_bn_input_shape[1]) + ",)";
            throw ngraph_error(err_msg.c_str());
        }
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
    if (this->m_training)
    {
        if (new_args.size() == 3)
        {
            return std::make_shared<BatchNorm>(
                m_epsilon, new_args.at(0), new_args.at(1), new_args.at(2));
        }
        else if (new_args.size() == 5)
        {
            return std::make_shared<BatchNorm>(m_epsilon,
                                               new_args.at(0),
                                               new_args.at(1),
                                               new_args.at(2),
                                               new_args.at(3),
                                               new_args.at(4),
                                               true);
        }
        else
        {
            throw ngraph_error("Incorrect number of new arguments");
        }
    }
    else
    {
        if (new_args.size() != 5)
        {
            throw ngraph_error("Incorrect number of new arguments");
        }
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

    if (get_input_shape(INPUT).size() != 4)
    {
        throw ngraph_error("Input expected to be a 4D tensor");
    }

    auto et = get_input_element_type(INPUT);
    const char* input_names[] = {"gamma", "beta", "input", "mean", "variance", "delta"};

    for (size_t i = 0; i < get_input_size(); i++)
    {
        if (get_input_element_type(i) != et)
        {
            auto err_msg = std::string("The element type of ") + input_names[i] +
                           " isn't equal to input data's type";
            throw ngraph_error(err_msg.c_str());
        }
    }

    Shape channel_shape{get_input_shape(INPUT).at(1)};

    for (size_t i = 0; i < get_input_size(); i++)
    {
        if (i == 2 || i == 5) // don't check input and delta
        {
            continue;
        }

        if (get_argument(i)->get_shape() != channel_shape)
        {
            auto err_msg = std::string("The shape of ") + input_names[i] +
                           " isn't equal to input channel's shape";
            throw ngraph_error(err_msg.c_str());
        }
    }

    if (get_input_shape(DELTA) != get_input_shape(INPUT))
    {
        throw ngraph_error("delta shape is expected to be equal to input shape");
    }

    set_output_type(0, get_input_element_type(INPUT), get_input_shape(INPUT));
    set_output_type(1, get_input_element_type(GAMMA), get_input_shape(GAMMA));
    set_output_type(2, get_input_element_type(BETA), get_input_shape(BETA));
}

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormBackprop::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 6)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
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
