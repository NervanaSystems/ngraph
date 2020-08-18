//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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
#include "ngraph/op/op.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        namespace gpu
        {
            class BatchNormTrainingWithStats : public ngraph::op::v0::BatchNormTraining
            {
            public:
                BatchNormTrainingWithStats(double eps,
                                           const Output<Node>& gamma,
                                           const Output<Node>& beta,
                                           const Output<Node>& input);

                void validate_and_infer_types() override;

                static constexpr NodeTypeInfo type_info{"BatchNormTrainingWithStats", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }

            protected:
                virtual std::shared_ptr<Node>
                    clone_with_new_inputs(const OutputVector& new_args) const override;
            };
        }
    }
}
