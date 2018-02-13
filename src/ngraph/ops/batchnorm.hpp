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

#pragma once

#include <memory>
#include "ngraph/node.hpp"
#include "ngraph/ops/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class BatchNorm : public RequiresTensorViewArgs
        {
        public:
            BatchNorm(double eps,
                      std::shared_ptr<Node> gamma,
                      std::shared_ptr<Node> beta,
                      std::shared_ptr<Node> input,
                      std::shared_ptr<Node> mean,
                      std::shared_ptr<Node> variance);

            const Shape& get_inputs_shape() const { return bn_input_shape; }
            const Shape& get_output_shape() const { return bn_output_shape; }
            const Shape& get_variance_shape() const { return bn_variance_shape; }
            const Shape& get_mean_shape() const { return bn_mean_shape; }
            const float get_eps_value() const { return epsilon; }
            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override;

        private:
            Shape bn_input_shape;
            Shape bn_output_shape;
            Shape bn_variance_shape;
            Shape bn_mean_shape;
            double epsilon;
        };
    }
}
