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

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/util.hpp"

#include <memory>

namespace ngraph
{
    namespace op
    {
        class BatchnormFprop: public RequiresTensorViewArgs
        {
        public:
            BatchnormFprop(std::shared_ptr<Node> eps,
                           std::shared_ptr<Node> gamma,
                           std::shared_ptr<Node> beta,
                           std::shared_ptr<Node> input,
                           Shape mean_shape,
                           Shape variance_shape,
                           Shape output_shape,
                           const element::Type& mean_et,
                           const element::Type& variance_et)
                        :RequiresTensorViewArgs("BatchnormFprop", {eps, gamma, beta, input})
                        ,mkl_output_shape(output_shape)
                        ,mkl_variance_shape(variance_shape)
                        ,mkl_mean_shape(mean_shape)
                        ,mkl_input_shape(input->get_shape())
                        ,mean_element_type(mean_et)
                        ,variance_element_type(variance_et)
            {
                add_output(input->get_element_type(), mkl_output_shape);
                add_output(mean_element_type, mkl_mean_shape);
                add_output(variance_element_type, mkl_variance_shape);
            }

        const Shape& get_inputs_shape() const{
            return mkl_input_shape;
        }

        const Shape& get_output_shape() const{
            return mkl_output_shape;
        }

        const Shape& get_variance_shape() const{
            return mkl_variance_shape;
        }

        const Shape& get_mean_shape() const{
            return mkl_mean_shape;
        }

        virtual std::shared_ptr<Node> copy_with_new_args(
            const std::vector<std::shared_ptr<Node>>& new_args) const override
        {
            if (new_args.size() != 4)
                throw ngraph_error("Incorrect number of new arguments");
            return std::make_shared<BatchnormFprop>(new_args.at(0),
                                                new_args.at(1),
                                                new_args.at(2),
                                                new_args.at(3),
                                                mkl_mean_shape,
                                                mkl_variance_shape,
                                                mkl_output_shape,
                                                mean_element_type,
                                                variance_element_type);
        }
        
        private:
                Shape mkl_input_shape;
                Shape mkl_output_shape;
                Shape mkl_variance_shape;
                Shape mkl_mean_shape;
                const element::Type& mean_element_type;
                const element::Type& variance_element_type;

        };
    }
}
