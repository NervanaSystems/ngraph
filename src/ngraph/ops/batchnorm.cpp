// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#include "ngraph/ops/batchnorm.hpp"
#include "ngraph/ops/constant.hpp"


ngraph::op::BatchnormFprop::BatchnormFprop(std::shared_ptr<ngraph::Node> eps,
               std::shared_ptr<ngraph::Node> gamma,
               std::shared_ptr<ngraph::Node> beta,
               std::shared_ptr<ngraph::Node> input,
               std::shared_ptr<ngraph::Node> mean,
               std::shared_ptr<ngraph::Node> variance,
               Shape output_shape,
               const element::Type& mean_et,
               const element::Type& variance_et)
            :RequiresTensorViewArgs("BatchnormFprop", {eps, gamma, beta, input, mean, variance})
            ,mkl_output_shape(output_shape)
            ,mkl_variance_shape(variance->get_shape())
            ,mkl_mean_shape(mean->get_shape())
            ,mkl_input_shape(input->get_shape())
            ,mean_element_type(mean_et)
            ,variance_element_type(variance_et)
{
        add_output(input->get_element_type(), mkl_output_shape);
	    //TODO add the sanity checkers for the inputs to bn

        auto eps_ptr = std::dynamic_pointer_cast<op::Constant>(eps);
        const float* p = reinterpret_cast<const float*>(eps_ptr->get_data_ptr());
        epsilon = *p;
}

std::shared_ptr<ngraph::Node> ngraph::op::BatchnormFprop::copy_with_new_args(
    const std::vector<std::shared_ptr<ngraph::Node>>& new_args) const 
{
    if (new_args.size() != 6)
        throw ngraph_error("Incorrect number of new arguments");
    return std::make_shared<BatchnormFprop>(new_args.at(0),
                                        new_args.at(1),
                                        new_args.at(2),
                                        new_args.at(3),
                                        new_args.at(4),
                                        new_args.at(5),
                                        mkl_output_shape,
                                        mean_element_type,
                                        variance_element_type);
}
