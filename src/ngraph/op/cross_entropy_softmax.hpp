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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/decollapsible.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/util.hpp"

#include <memory>

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise CrossEntropySoftMax operation.
        ///
        class CrossEntropySoftMax : public util::Decollapsible
        {
        public:
            /// \brief Constructs a CrossEntropySoftMax operation.
            ///
            /// \param y is probabilities for each class and every sample
            /// \param t is class labels for every sample
            /// \param old_cross_entropy for fissioning CrossEntropySoftMax back into Sum(SoftMax)
            CrossEntropySoftMax(std::shared_ptr<ngraph::Node> y,
                                std::shared_ptr<ngraph::Node> t,
                                std::shared_ptr<Node> old_cross_entropy,
                                size_t axis);

            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override
            {
                throw ngraph_error("Uncopyable");
            }
        };
    }
}
