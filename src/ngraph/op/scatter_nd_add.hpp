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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Add updates to slices from inputs addressed by indices
        class ScatterNDAdd : public Op
        {
        public:
            /// \param inputs Tensor
            /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
            /// \param updates Tensor: Must have same type as inputs
            ScatterNDAdd(const std::shared_ptr<Node>& inputs,
                         const std::shared_ptr<Node>& indices,
                         const std::shared_ptr<Node>& updates)
                : Op("ScatterNDAdd", check_single_output_args({inputs, indices, updates}))
            {
                constructor_validate_and_infer_types();
            }

            void validate_and_infer_types() override;

            void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) override
            {
                throw ngraph_error("Not yet implemented");
            }

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
