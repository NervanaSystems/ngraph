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
                           Shape output_shape)
                        :RequiresTensorViewArgs("BatchnormFprop", {eps, gamma, beta, input})
                        ,mkl_output_shape (output_shape)
            {
                add_output(input->get_element_type(), mkl_output_shape);
            }
            
        const Shape& get_weights_shape() const{
            return mkl_weights_shape;
        }

        const Shape& get_inputs_shape() const{
            return mkl_input_shape;
        }

        const Shape& get_output_shape() const{
            return mkl_output_shape;
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
                                                mkl_output_shape);
        }
        
        private:
                Shape mkl_input_shape;
                Shape mkl_output_shape;
                Shape mkl_weights_shape; // MKLDNN expects gamma and weights to be stacked in a single tensor;

        };
    }
}
