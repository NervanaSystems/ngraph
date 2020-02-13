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
        namespace v0
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Gather", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Gather() = default;
                /// \param params The tensor from which slices are gathered
                /// \param indices Index tensor: Data type must be `element::i32` or `element::i64`
                /// \param axis Axis in params to gather
                Gather(const Output<Node>& params, const Output<Node>& indices, size_t axis = 0);

                void validate_and_infer_types() override;

                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;

                size_t get_axis() const { return m_axis; }
                void set_axis(size_t axis) { m_axis = axis; }
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

            protected:
                size_t m_axis;
            };
        }

        namespace v1
        {
            /// \brief Gather slices from axis of params according to indices
            class NGRAPH_API Gather : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"Gather", 1};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                Gather() = default;
                /// \param params The tensor from which slices are gathered
                /// \param indices Tensor with indexes to gather
                /// \param axis The tensor is a dimension index to gather data from
                Gather(const Output<Node>& params,
                       const Output<Node>& indices,
                       const Output<Node>& axis);

                int64_t get_axis() const;

                void validate_and_infer_types() override;

                void generate_adjoints(autodiff::Adjoints& adjoints,
                                       const OutputVector& deltas) override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
            };
        }

        // latest stable opset version
        using v0::Gather;
    }
}
