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

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        class SliceInput : public util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs an empty SliceInput operation
            SliceInput() = default;

            // start, stride, part_size, and end could be inputs rather than attributes
            // axis could be a vector of axes for a multi-dimensional slice walk

            /// \brief Describes how to slice an input for iteration
            /// \param value The value to be sliced
            /// \param start First index on axis of the slicing
            /// \param stride Stepping of the slice
            /// \param part_size Size of the slice on axis
            /// \param end The last index on axis of the slicing
            /// \param axis The axis to slice along
            SliceInput(const Output<Node>& value,
                       std::ptrdiff_t start,
                       std::ptrdiff_t stride,
                       std::ptrdiff_t part_size,
                       std::ptrdiff_t end,
                       std::ptrdiff_t axis);

            void validate_and_infer_types() override;
            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
            NodeVector decompose_op() const override;

            std::ptrdiff_t get_axis() const;
            void set_axis(std::ptrdiff_t axis);

            std::ptrdiff_t get_start() const;
            void set_start(std::ptrdiff_t start);

            std::ptrdiff_t get_stride() const;
            void set_stride(std::ptrdiff_t stride);

            std::ptrdiff_t get_part_size() const;
            void set_part_size(std::ptrdiff_t part_size);

            std::ptrdiff_t get_end() const;
            void set_end(std::ptrdiff_t end);

        protected:
            std::ptrdiff_t m_start;
            std::ptrdiff_t m_stride;
            std::ptrdiff_t m_part_size;
            std::ptrdiff_t m_end;
            std::ptrdiff_t m_axis;
        };
    }
}
