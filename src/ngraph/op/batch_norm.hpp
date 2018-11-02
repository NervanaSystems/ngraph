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

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class BatchNormTraining : public Op
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
            BatchNormTraining(double eps,
                              std::shared_ptr<Node> gamma,
                              std::shared_ptr<Node> beta,
                              std::shared_ptr<Node> input);

            // In this version of BatchNorm:
            //
            // MEAN AND VARIANCE: provided by the 'mean' and 'variance' parameters.
            //
            // OUTPUT VALUE: a single tensor with the normalized value of 'input'.
            // mean and variance will also be updated inplace
            //
            // AUTODIFF SUPPORT:
            //   'generate_adjoints(...)' works as expected.
            //
            // SHAPE DETAILS:
            //   gamma:    must have rank 1, with the same span as input's channel axis.
            //   beta:     must have rank 1, with the same span as input's channel axis.
            //   input:    must have rank >= 2. The second dimension represents the channel axis and
            //             must have a span of at least 1.
            //   mean:     must have rank 1, with the same span as input's channel axis.
            //   variance: must have rank 1, with the same span as input's channel axis.
            //   output:   shall have the same shape as 'input'.
            BatchNormTraining(double eps,
                              std::shared_ptr<ngraph::Node> gamma,
                              std::shared_ptr<ngraph::Node> beta,
                              std::shared_ptr<ngraph::Node> input,
                              std::shared_ptr<ngraph::Node> mean,
                              std::shared_ptr<ngraph::Node> variance);

            void validate_and_infer_types() override;

            double get_eps_value() const { return m_epsilon; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            static constexpr size_t INPUT_GAMMA = 0;
            static constexpr size_t INPUT_BETA = 1;
            static constexpr size_t INPUT_DATA = 2;

            double m_epsilon;
        };

        class BatchNormInference : public Op
        {
        public:
            // In this version of BatchNorm:
            //
            // MEAN AND VARIANCE: provided by the 'mean' and 'variance' parameters.
            //
            // OUTPUT VALUE: a single tensor with the normalized value of 'input'.
            //
            // AUTODIFF SUPPORT:
            //   - 'generate_adjoints(...) may throw an exception.
            //
            // SHAPE DETAILS:
            //   gamma:    must have rank 1, with the same span as input's channel axis.
            //   beta:     must have rank 1, with the same span as input's channel axis.
            //   input:    must have rank >= 2. The second dimension represents the channel axis and
            //             must have a span of at least 1.
            //   mean:     must have rank 1, with the same span as input's channel axis.
            //   variance: must have rank 1, with the same span as input's channel axis.
            //   output:   shall have the same shape as 'input'.
            BatchNormInference(double eps,
                               std::shared_ptr<ngraph::Node> gamma,
                               std::shared_ptr<ngraph::Node> beta,
                               std::shared_ptr<ngraph::Node> input,
                               std::shared_ptr<ngraph::Node> mean,
                               std::shared_ptr<ngraph::Node> variance);

            void validate_and_infer_types() override;

            double get_eps_value() const { return m_epsilon; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override
            {
                throw ngraph_error("Invalid operation");
            }

        private:
            static constexpr size_t INPUT_GAMMA = 0;
            static constexpr size_t INPUT_BETA = 1;
            static constexpr size_t INPUT_DATA = 2;
            static constexpr size_t INPUT_MEAN = 3;
            static constexpr size_t INPUT_VARIANCE = 4;

            double m_epsilon;
        };

        class BatchNormTrainingBackprop : public Op
        {
        public:
            BatchNormTrainingBackprop(double eps,
                                      std::shared_ptr<Node> gamma,
                                      std::shared_ptr<Node> beta,
                                      std::shared_ptr<Node> input,
                                      std::shared_ptr<Node> mean,
                                      std::shared_ptr<Node> variance,
                                      std::shared_ptr<Node> delta);

            void validate_and_infer_types() override;

            double get_eps_value() const { return m_epsilon; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        private:
            static constexpr size_t INPUT_GAMMA = 0;
            static constexpr size_t INPUT_BETA = 1;
            static constexpr size_t INPUT_DATA = 2;
            static constexpr size_t INPUT_MEAN = 3;
            static constexpr size_t INPUT_VARIANCE = 4;
            static constexpr size_t INPUT_DELTA = 5;

            double m_epsilon;
        };
    }
}
