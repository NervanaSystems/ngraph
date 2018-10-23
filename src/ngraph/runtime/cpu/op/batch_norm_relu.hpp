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
        class BatchNormTrainingRelu : public Op
        {
        public:
            BatchNormTrainingRelu(double eps,
                                  std::shared_ptr<Node> gamma,
                                  std::shared_ptr<Node> beta,
                                  std::shared_ptr<Node> input);

            double get_eps_value() const { return m_epsilon; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            enum
            {
                GAMMA,
                BETA,
                INPUT,
                MEAN,
                VARIANCE,
                DELTA
            };

        private:
            double m_epsilon;
        };

        class BatchNormInferenceRelu : public Op
        {
        public:
            BatchNormInferenceRelu(double eps,
                                   std::shared_ptr<ngraph::Node> gamma,
                                   std::shared_ptr<ngraph::Node> beta,
                                   std::shared_ptr<ngraph::Node> input,
                                   std::shared_ptr<ngraph::Node> mean,
                                   std::shared_ptr<ngraph::Node> variance);

            double get_eps_value() const { return m_epsilon; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            enum
            {
                GAMMA,
                BETA,
                INPUT,
                MEAN,
                VARIANCE,
                DELTA
            };

        private:
            double m_epsilon;
        };
    }
}
