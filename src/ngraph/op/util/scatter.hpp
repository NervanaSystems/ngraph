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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Applies updates to slices from inputs addressed by indices
            class NGRAPH_API Scatter : public Op
            {
            public:
                Scatter() = default;

                /// \param inputs Tensor
                /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
                /// \param updates Tensor: Must have same type as inputs
                Scatter(const Output <Node> &inputs,
                        const Output <Node> &indices,
                        const Output <Node> &updates,
                        const int32_t axis = 0)
                        : Op({inputs, indices, updates}), m_axis(axis)
                {
                    constructor_validate_and_infer_types();
                }

                void validate_and_infer_types() override;

                void generate_adjoints(autodiff::Adjoints & /* adjoints */,
                                       const OutputVector & /* deltas */) override
                {
                    throw ngraph_error("Not yet implemented");
                }
                int32_t get_axis() const;
            protected:
                int32_t m_axis;
            };
        }
    }
}
