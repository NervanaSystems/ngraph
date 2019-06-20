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
        /// \brief Gather slices from params with shapes given by indices
        class GatherND : public Op
        {
        public:
            /// \param params The tensor from which slices are gathered
            /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
            GatherND(const std::shared_ptr<Node>& params, const std::shared_ptr<Node>& indices)
                : Op("GatherND", check_single_output_args({params, indices}))
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
