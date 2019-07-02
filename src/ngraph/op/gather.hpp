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
        /// \brief Gather slices from axis of params according to indices
        class Gather : public Op
        {
        public:
            /// \param params The tensor from which slices are gathered
            /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
            /// \param axis Axis in params to gather
            Gather(const std::shared_ptr<Node>& params,
                   const std::shared_ptr<Node>& indices,
                   size_t axis = 0)
                : Op("Gather", check_single_output_args({params, indices}))
                , m_axis(axis)
            {
                constructor_validate_and_infer_types();
            }

            void validate_and_infer_types() override;

            void generate_adjoints(autodiff::Adjoints& adjoints, const NodeVector& deltas) override
            {
                throw ngraph_error("Not yet implemented");
            }

            size_t get_axis() const { return m_axis; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            size_t m_axis;
        };
    }
}
