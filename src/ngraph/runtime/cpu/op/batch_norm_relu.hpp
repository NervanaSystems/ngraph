//*****************************************************************************
// Copyright 2017-2019 Intel Corporation
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
#include "ngraph/op/op.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class BatchNormTrainingRelu : public Op
        {
        public:
            CPU_BACKEND_API
            static constexpr NodeTypeInfo type_info{"BatchNormTrainingRelu", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            CPU_BACKEND_API BatchNormTrainingRelu(double eps,
                                                  const Output<Node>& gamma,
                                                  const Output<Node>& beta,
                                                  const Output<Node>& input);

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
            CPU_BACKEND_API
            static constexpr NodeTypeInfo type_info{"BatchNormInferenceRelu", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            BatchNormInferenceRelu(double eps,
                                   const Output<ngraph::Node>& gamma,
                                   const Output<ngraph::Node>& beta,
                                   const Output<ngraph::Node>& input,
                                   const Output<ngraph::Node>& mean,
                                   const Output<ngraph::Node>& variance);

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
