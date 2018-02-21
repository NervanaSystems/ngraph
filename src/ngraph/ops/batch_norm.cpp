/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/get_output_element.hpp"

ngraph::op::BatchNorm::BatchNorm(double eps,
                                 std::shared_ptr<ngraph::Node> gamma,
                                 std::shared_ptr<ngraph::Node> beta,
                                 std::shared_ptr<ngraph::Node> input,
                                 std::shared_ptr<ngraph::Node> mean,
                                 std::shared_ptr<ngraph::Node> variance)
    : RequiresTensorViewArgs("BatchNorm", {gamma, beta, input, mean, variance})
    , m_bn_input_shape(input->get_shape())
    , m_bn_variance_shape(variance->get_shape())
    , m_bn_mean_shape(mean->get_shape())
    , m_epsilon(eps)
{
    add_output(input->get_element_type(), m_bn_input_shape);

    if (m_bn_input_shape.size() < 2)
    {
        throw ngraph_error("input tensor to batchnorm much have tensor of atleast rank 2");
    }

    if (m_bn_input_shape[1] == 0)
    {
        throw ngraph_error(
            "input tensor must have atleast one channel axis for batch normalization");
    }

    if ((m_bn_mean_shape.size() != 1) && (m_bn_variance_shape.size() != 1) &&
        (gamma->get_shape().size() != 1) && (beta->get_shape().size() != 1))
    {
        throw ngraph_error("gamma, beta, mean, variance shoud have all rank 1");
    }

    // assuming input shape (N, C, H, W), check if the size of mean and
    // variance are equal to channel axis
    if (mean->get_shape()[0] != m_bn_input_shape[1])
    {
        throw ngraph_error("mean size is not equal to input channel size");
    }

    if (variance->get_shape()[0] != m_bn_input_shape[1])
    {
        throw ngraph_error("variance size is not equal to input channel size");
    }

    if (variance->get_shape().size() != mean->get_shape().size())
    {
        throw ngraph_error("mean and variance rank does not match");
    }

    if (gamma->get_shape().size() != beta->get_shape().size())
    {
        throw ngraph_error("gamma and beta rank does not match");
    }

    if (input->get_element_type() != mean->get_element_type())
    {
        throw ngraph_error("input tensor and mean element type does not match");
    }

    if (input->get_element_type() != variance->get_element_type())
    {
        throw ngraph_error("input tensor and variance element type does not match");
    }

    if (gamma->get_element_type() != beta->get_element_type())
    {
        throw ngraph_error("gamma and beta element type does not match");
    }
}

std::shared_ptr<ngraph::Node> ngraph::op::BatchNorm::copy_with_new_args(
    const std::vector<std::shared_ptr<ngraph::Node>>& new_args) const
{
    if (new_args.size() != 5)
        throw ngraph_error("Incorrect number of new arguments");
    return std::make_shared<BatchNorm>(
        m_epsilon, new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
}

ngraph::op::BatchNormBackprop::BatchNormBackprop(double eps,
	std::shared_ptr<ngraph::Node> gamma,
	std::shared_ptr<ngraph::Node> beta,
	std::shared_ptr<ngraph::Node> input,
	std::shared_ptr<ngraph::Node> mean,
	std::shared_ptr<ngraph::Node> variance,
	std::shared_ptr<ngraph::Node> delta)
	: RequiresTensorViewArgs("BatchNormBackprop", { gamma, beta, input, mean, variance, delta })
	, epsilon(eps)

{
	if (input->get_shape().size() != 4)
	{
		throw ngraph_error("Input expected to be a 4D tensor");
	}

	auto et = input->get_element_type();
	const char* input_names[] = { "gamma", "beta", "input", "mean", "variance", "delta" };


	for (size_t i = 0; i < get_input_size(); i++)
	{
		if (get_input_op(i)->get_element_type() != et)
		{
			auto err_msg = std::string("The element type of ") + input_names[i] + " isn't equal to input data's type";
			throw ngraph_error(err_msg.c_str());
		}
	}

	Shape channel_shape{input->get_shape().at(1)};

	for (size_t i = 0; i < get_input_size(); i++)
	{
		if (i == 2 || i == 5) //don't check input and delta
		{
			continue;
		}

		if (get_input_op(i)->get_shape() != channel_shape)
		{
			auto err_msg = std::string("The shape of ") + input_names[i] + " isn't equal to input channel's shape";
			throw ngraph_error(err_msg.c_str());
		}
	}

	if (delta->get_shape() != input->get_shape())
	{
		throw ngraph_error("delta shape is expected to be equal to input shape");
	}

	add_output(input->get_element_type(), input->get_shape());
    add_output(gamma->get_element_type(), gamma->get_shape());
    add_output(beta->get_element_type(), beta->get_shape());
}

std::shared_ptr<ngraph::Node> ngraph::op::BatchNormBackprop::copy_with_new_args(
	const std::vector<std::shared_ptr<ngraph::Node>>& new_args) const
{
	if (new_args.size() != 6)
    {
		throw ngraph_error("Incorrect number of new arguments");
    }
	return std::make_shared<op::BatchNormBackprop>(
		epsilon, new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4), new_args.at(5));
}


void ngraph::op::BatchNorm::generate_adjoints(autodiff::Adjoints& adjoints,
	const std::shared_ptr<Node>& delta)
{
	auto gamma = get_input_op(0);
	auto beta = get_input_op(1);
	auto input = get_input_op(2); 
	auto mean = get_input_op(3);
	auto variance = get_input_op(4);
    auto bbn = std::make_shared<op::BatchNormBackprop>(get_eps_value(), gamma, beta, input, mean, variance, delta);
    auto dinput = std::make_shared<op::GetOutputElement>(bbn, 0);
    auto dgamma = std::make_shared<op::GetOutputElement>(bbn, 1);
    auto dbeta = std::make_shared<op::GetOutputElement>(bbn, 2);

	adjoints.add_delta(input, dinput);
    adjoints.add_delta(gamma, dgamma);
    adjoints.add_delta(beta, dbeta);
}
