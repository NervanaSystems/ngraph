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

#include "ngraph/runtime/gpu/op/batch_norm.hpp"
#include "ngraph/node.hpp"
#include "ngraph/validation_util.hpp"

constexpr ngraph::NodeTypeInfo ngraph::op::gpu::BatchNormTrainingWithStats::type_info;

ngraph::op::gpu::BatchNormTrainingWithStats::BatchNormTrainingWithStats(
    double eps,
    std::shared_ptr<ngraph::Node> gamma,
    std::shared_ptr<ngraph::Node> beta,
    std::shared_ptr<ngraph::Node> input)
    : ngraph::op::BatchNormTraining(eps, gamma, beta, input)
{
    constructor_validate_and_infer_types();
}

void ngraph::op::gpu::BatchNormTrainingWithStats::validate_and_infer_types()
{
    element::Type result_et;
    PartialShape result_batch_shape;
    PartialShape result_channel_shape;

    set_output_size(5);
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
    // saved batch mean
    set_output_type(3, result_et, result_channel_shape);
    // saved batch inverse variance
    set_output_type(4, result_et, result_channel_shape);
}

std::shared_ptr<ngraph::Node> ngraph::op::gpu::BatchNormTrainingWithStats::copy_with_new_args(
    const NodeVector& new_args) const
{
    check_new_args_count(this, new_args);
    return std::make_shared<ngraph::op::gpu::BatchNormTrainingWithStats>(
        get_eps_value(), new_args.at(0), new_args.at(1), new_args.at(2));
}
