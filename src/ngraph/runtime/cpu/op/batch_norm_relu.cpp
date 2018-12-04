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

#include "ngraph/runtime/cpu/op/batch_norm_relu.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/get_output_element.hpp"

ngraph::op::BatchNormTrainingRelu::BatchNormTrainingRelu(double eps,
                                                         std::shared_ptr<ngraph::Node> gamma,
                                                         std::shared_ptr<ngraph::Node> beta,
                                                         std::shared_ptr<ngraph::Node> input)
    : Op("BatchNormTrainingRelu", check_single_output_args({gamma, beta, input}))
    , m_epsilon(eps)
{
    constructor_validate_and_infer_types();

    auto bn_input_shape = get_input_shape(INPUT);

    if (bn_input_shape.size() != 4)
    {
        throw ngraph_error("input tensor to batchnorm must have rank 4");
    }

    auto channel_shape = Shape{bn_input_shape.at(1)};

    if (bn_input_shape[1] == 0)
    {
        throw ngraph_error(
            "input tensor must have at least one channel axis for batch normalization");
    }

    auto et = input->get_element_type();
    const char* input_names[] = {"gamma", "beta"};

    for (size_t i = 0; i < 2; i++)
    {
        if (get_argument(i)->get_element_type() != et)
        {
            auto err_msg = std::string("The element type of ") + input_names[i] +
                           " isn't equal to input data's type";
            throw ngraph_error(err_msg.c_str());
        }
    }

    if ((gamma->get_shape().size() != 1) || (beta->get_shape().size() != 1))
    {
        throw ngraph_error("gamma and beta should have rank 1");
    }

    if (gamma->get_shape().size() != beta->get_shape().size())
    {
        throw ngraph_error("gamma and beta rank does not match");
    }

    if (gamma->get_element_type() != beta->get_element_type())
    {
        throw ngraph_error("gamma and beta element type does not match");
    }

    set_output_size(3);
    set_output_type(0, input->get_element_type(), bn_input_shape);
    set_output_type(1, input->get_element_type(), channel_shape);
    set_output_type(2, input->get_element_type(), channel_shape);
}

ngraph::op::BatchNormInferenceRelu::BatchNormInferenceRelu(double eps,
                                                           std::shared_ptr<ngraph::Node> gamma,
                                                           std::shared_ptr<ngraph::Node> beta,
                                                           std::shared_ptr<ngraph::Node> input,
                                                           std::shared_ptr<ngraph::Node> mean,
                                                           std::shared_ptr<ngraph::Node> variance)
    : Op("BatchNormInferenceRelu", check_single_output_args({gamma, beta, input, mean, variance}))
    , m_epsilon(eps)
{
    constructor_validate_and_infer_types();
    auto bn_input_shape = get_input_shape(INPUT);
    if (bn_input_shape.size() != 4)
    {
        throw ngraph_error("input tensor to batchnorm must have rank 4");
    }

    if (bn_input_shape[1] == 0)
    {
        throw ngraph_error(
            "input tensor must have at least one channel axis for batch normalization");
    }

    auto et = input->get_element_type();
    const char* input_names[] = {"gamma", "beta"};

    for (size_t i = 0; i < 2; i++)
    {
        if (get_argument(i)->get_element_type() != et)
        {
            auto err_msg = std::string("The element type of ") + input_names[i] +
                           " isn't equal to input data's type";
            throw ngraph_error(err_msg.c_str());
        }
    }

    if ((gamma->get_shape().size() != 1) || (beta->get_shape().size() != 1))
    {
        throw ngraph_error("gamma and beta should have rank 1");
    }

    if (gamma->get_shape().size() != beta->get_shape().size())
    {
        throw ngraph_error("gamma and beta rank does not match");
    }

    if (gamma->get_element_type() != beta->get_element_type())
    {
        throw ngraph_error("gamma and beta element type does not match");
    }

    set_output_type(0, input->get_element_type(), bn_input_shape);
}

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormTrainingRelu::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() == 3)
    {
        return std::make_shared<BatchNormTrainingRelu>(
            m_epsilon, new_args.at(0), new_args.at(1), new_args.at(2));
    }
    else
    {
        throw ngraph_error("BatchNormRelu: Incorrect number of new arguments");
    }
}

std::shared_ptr<ngraph::Node>
    ngraph::op::BatchNormInferenceRelu::copy_with_new_args(const NodeVector& new_args) const
{
    if (new_args.size() != 5)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return std::make_shared<BatchNormInferenceRelu>(
        m_epsilon, new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}
