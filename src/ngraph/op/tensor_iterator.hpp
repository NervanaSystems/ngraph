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
#include "ngraph/op/parameter.hpp"

namespace ngraph
{
    namespace op
    {
        struct InputSliceDescription
        {
            Output<Node> m_value;
            std::ptrdiff_t m_start;
            std::ptrdiff_t m_stride;
            std::ptrdiff_t m_part_size;
            std::ptrdiff_t m_end;
            std::ptrdiff_t m_axis;
        };

        struct BodyConnection
        {
            Output<Node> m_initial_value;
            Output<Node> m_body_value;
        };

        struct OutputConcatDescription
        {
            Output<Node> m_body_value;
            std::ptrdiff_t m_start;
            std::ptrdiff_t m_stride;
            std::ptrdiff_t m_part_size;
            std::ptrdiff_t m_end;
            std::ptrdiff_t m_axis;
        };

        struct OutputSliceDescription
        {
            Output<Node> m_body_value;
            int64_t m_index;
        };

        /// \brief  Iterate a body over tensors, accumulating into tensors
        class TensorIterator : public util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
        public:
            /// \brief Indicate that a body parameter comes from slices of a value
            /// \param parameter The parameter to receive the slices
            /// \param value The value to be sliced
            /// \param start First index on axis of the slicing
            /// \param stride Stepping of the slice
            /// \param part_size Size of the slice on axis
            /// \param end The last index on axis of the slicing
            /// \param axis The axis to slice along
            void set_sliced_input(const std::shared_ptr<Parameter>& parameter,
                                  const Output<Node>& value,
                                  int64_t start,
                                  int64_t stride,
                                  int64_t part_size,
                                  int64_t end,
                                  int64_t axis);
            /// \brief Indicates that a body parameter has an initial value in the first iteration
            /// and computed value thereafter
            /// \param initial_value Value for the parameter in first iteration
            /// \param successive_value Value for the parameter in successive iterations. The
            /// value is what is active in the most recent completed iteration.
            void set_initialized_input(const std::shared_ptr<Parameter>& body_parameter,
                                       const Output<Node>& initial_value,
                                       const Output<Node>& successive_value);
            /// \brief Gets a value for a particular iteration point
            /// \param The value
            /// \param iteration. Negative values are from the last iteration.
            Output<Node> get_iter_value(const Output<Node>& body_value, int64_t iteration);
            /// \brief Concatenates slices from all iterations
            /// \param value The value supplying slice values from each iteration.
            /// \param start First index on axis of the slicing
            /// \param stride Stepping of the slice
            /// \param part_size Size of the slice on axis
            /// \param end The last index on axis of the slicing
            /// \param axis The axis to slice along
            Output<Node> get_concatenated_slices(const Output<Node>& value,
                                                 int64_t start,
                                                 int64_t stride,
                                                 int64_t part_size,
                                                 int64_t end,
                                                 int64_t axis);

        private:
        };
    }
}
