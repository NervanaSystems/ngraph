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

#include "ngraph/runtime/gpu/op/batch_norm.hpp"

ngraph::op::gpu::BatchNormTrainingWithStats::BatchNormTrainingWithStats(
    double eps,
    std::shared_ptr<ngraph::Node> gamma,
    std::shared_ptr<ngraph::Node> beta,
    std::shared_ptr<ngraph::Node> input)
    : ngraph::op::BatchNormTraining(eps, gamma, beta, input)
{
    auto output_index = get_output_size();
    set_output_size(output_index + 2);
    Shape channel_shape{input->get_shape()[1]};
    // saved batch mean
    set_output_type(output_index++, input->get_element_type(), channel_shape);
    // saved batch inverse variance
    set_output_type(output_index++, input->get_element_type(), channel_shape);
}

ngraph::op::gpu::BatchNormTrainingWithStats::BatchNormTrainingWithStats(
    double eps,
    std::shared_ptr<ngraph::Node> gamma,
    std::shared_ptr<ngraph::Node> beta,
    std::shared_ptr<ngraph::Node> input,
    std::shared_ptr<ngraph::Node> mean,
    std::shared_ptr<ngraph::Node> variance,
    bool training)
    : ngraph::op::BatchNormTraining(eps, gamma, beta, input, mean, variance)
{
    auto output_index = get_output_size();
    set_output_size(output_index + 2);
    Shape channel_shape{input->get_shape()[1]};
    // saved batch mean
    set_output_type(output_index++, input->get_element_type(), channel_shape);
    // saved batch inverse variance
    set_output_type(output_index++, input->get_element_type(), channel_shape);
}

std::shared_ptr<ngraph::Node> ngraph::op::gpu::BatchNormTrainingWithStats::copy_with_new_args(
    const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<ngraph::op::gpu::BatchNormTrainingWithStats>(
        get_eps_value(), new_args.at(0), new_args.at(1), new_args.at(2));
}
