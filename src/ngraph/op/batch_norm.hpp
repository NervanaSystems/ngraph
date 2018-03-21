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
#include "ngraph/node_vector.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class BatchNorm : public util::RequiresTensorViewArgs
        {
        public:
            BatchNorm(double eps,
                      std::shared_ptr<Node> gamma,
                      std::shared_ptr<Node> beta,
                      std::shared_ptr<Node> input);

            const Shape& get_inputs_shape() const { return m_bn_input_shape; }
            const Shape& get_variance_shape() const { return m_bn_variance_shape; }
            const Shape& get_mean_shape() const { return m_bn_mean_shape; }
            double get_eps_value() const { return m_epsilon; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;

        private:
            Shape m_bn_input_shape;
            Shape m_bn_variance_shape;
            Shape m_bn_mean_shape;
            double m_epsilon;
        };

        class BatchNormBackprop : public util::RequiresTensorViewArgs
        {
        public:
            BatchNormBackprop(double eps,
                              std::shared_ptr<Node> gamma,
                              std::shared_ptr<Node> beta,
                              std::shared_ptr<Node> input,
                              std::shared_ptr<Node> mean,
                              std::shared_ptr<Node> variance,
                              std::shared_ptr<Node> delta);

            double get_eps_value() const { return epsilon; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            double epsilon;
        };
    }
}
