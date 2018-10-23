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

#include "ngraph/op/batch_norm.hpp"

#include "ngraph/node.hpp"
#include "ngraph/node_vector.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        namespace gpu
        {
            class BatchNormTrainingWithStats : public ngraph::op::BatchNormTraining
            {
            public:
                BatchNormTrainingWithStats(double eps,
                                           std::shared_ptr<Node> gamma,
                                           std::shared_ptr<Node> beta,
                                           std::shared_ptr<Node> input);

                BatchNormTrainingWithStats(double eps,
                                           std::shared_ptr<ngraph::Node> gamma,
                                           std::shared_ptr<ngraph::Node> beta,
                                           std::shared_ptr<ngraph::Node> input,
                                           std::shared_ptr<ngraph::Node> mean,
                                           std::shared_ptr<ngraph::Node> variance,
                                           bool training = false);

            protected:
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }
    }
}
