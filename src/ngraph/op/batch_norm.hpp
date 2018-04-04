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
            // In this version of BatchNorm:
            //
            // MEAN AND VARIANCE: computed directly from the content of 'input'.
            //
            // OUTPUT VALUE: A tuple with the following structure:
            //   [0] - The normalization of 'input'.
            //   [1] - The per-channel means of (pre-normalized) 'input'.
            //   [2] - The per-channel variances of (pre-normalized) 'input'.
            //
            // AUTODIFF SUPPORT: yes: 'generate_adjoints(...)' works as expected.
            //
            // SHAPE DETAILS:
            //   gamma:     must have rank 1, with the same span as input's channel axis.
            //   beta:      must have rank 1, with the same span as input's channel axis.
            //   input:     must have rank >= 2.  The second dimension represents the channel axis
            //              and must have a span of at least 1.
            //   output[0]: shall have the same shape as 'input'.
            //   output[1]: shall have rank 1, with the same span as input's channel axis.
            //   output[2]: shall have rank 1, with the same span as input's channel axis.
            BatchNorm(double eps,
                      std::shared_ptr<Node> gamma,
                      std::shared_ptr<Node> beta,
                      std::shared_ptr<Node> input);

            // In this version of BatchNorm:
            //
            // MEAN AND VARIANCE: provided by the 'mean' and 'variance' parameters.
            //
            // OUTPUT VALUE: a single tensor with the normalized value of 'input'.
            //
            // AUTODIFF SUPPORT:
            //   - when 'training' is true, yes: 'generate_adjoints(...)' works as expected.
            //   - when 'training' is false, no: 'generate_adjoints(...) may throw an exception.
            //
            // SHAPE DETAILS:
            //   gamma:    must have rank 1, with the same span as input's channel axis.
            //   beta:     must have rank 1, with the same span as input's channel axis.
            //   input:    must have rank >= 2. The second dimension represents the channel axis and
            //             must have a span of at least 1.
            //   mean:     must have rank 1, with the same span as input's channel axis.
            //   variance: must have rank 1, with the same span as input's channel axis.
            //   output:   shall have the same shape as 'input'.
            BatchNorm(double eps,
                      std::shared_ptr<ngraph::Node> gamma,
                      std::shared_ptr<ngraph::Node> beta,
                      std::shared_ptr<ngraph::Node> input,
                      std::shared_ptr<ngraph::Node> mean,
                      std::shared_ptr<ngraph::Node> variance,
                      bool training = false);

            const Shape& get_inputs_shape() const { return m_bn_input_shape; }
            const Shape& get_variance_shape() const { return m_bn_variance_shape; }
            const Shape& get_mean_shape() const { return m_bn_mean_shape; }
            double get_eps_value() const { return m_epsilon; }
            bool get_training_flag() const { return m_training; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            Shape m_bn_input_shape;
            Shape m_bn_variance_shape;
            Shape m_bn_mean_shape;
            double m_epsilon;
            bool m_training;
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
