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
                           std::shared_ptr<Node> mean,
                           std::shared_ptr<Node> variance,
                           Shape output_shape,
                           const element::Type& mean_et,
                           const element::Type& variance_et);

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

        const float get_eps_value() const{
            return epsilon;
        }

        virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override; 

        private:
                Shape mkl_input_shape;
                Shape mkl_output_shape;
                Shape mkl_variance_shape;
                Shape mkl_mean_shape;
                const element::Type& mean_element_type;
                const element::Type& variance_element_type;
                float epsilon;

        };
    }
}
