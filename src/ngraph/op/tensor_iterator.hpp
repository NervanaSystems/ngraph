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
        struct SliceInput
        {
            /// \brief Describes how to slice an input for iteration
            /// \param argument_position Selects the argument to slice
            /// \param body_parameter Body parameter that receives the sliced input
            /// \param axis The axis to slice along
            /// \param start First index on axis of the slicing
            /// \param stride Stepping of the slice
            /// \param part_size Size of the slice on axis
            /// \param end The last index on axis of the slicing
            SliceInput(size_t argument_position,
                       const std::shared_ptr<Parameter>& body_parameter,
                       ssize_t axis,
                       ssize_t start,
                       ssize_t stride,
                       ssize_t part_size,
                       ssize_t end);
            size_t m_argument_position;
            std::shared_ptr<Parameter> m_body_parameter;
            ssize_t m_axis;
            ssize_t m_start;
            ssize_t m_stride;
            ssize_t m_part_size;
            ssize_t m_end;
        };

        struct SliceOutput
        {
            /// \brief Describes how to construct an output from slices
            /// \param value The value that generates the output slice
            /// \param result_position Selects the result to construct
            /// \param axis The axis to slice along
            /// \param start First index on axis of the slicing
            /// \param stride Stepping of the slice
            /// \param part_size Size of the slice on axis
            /// \param end The last index on axis of the slicing
            SliceOutput(const Output<Node>& value,
                        size_t result_position,
                        ssize_t axis,
                        ssize_t start,
                        ssize_t stride,
                        ssize_t part_size,
                        ssize_t end);

            Output<Node> m_value;
            size_t m_result_position;
            ssize_t m_axis;
            ssize_t m_start;
            ssize_t m_stride;
            ssize_t m_part_size;
            ssize_t m_end;
        };

        /// \brief  Iterate a body over tensors, accumulating into tensors
        class TensorIterator : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs an empty TensorIterator operation
            TensorIterator();

            TensorIterator(const OutputVector& inputs,
                           const ParameterVector& body_parameters,
                           const OutputVector& initial_body_arguments,
                           const OutputVector& body_arguments,
                           const OutputVector& outputs,
                           std::vector<SliceInput> slice_inputs,
                           std::vector<SliceOutput> slice_outputs);

            const ParameterVector& get_body_parameters() const;
            ParameterVector& get_body_parameters();
            void set_body_parameters(const ParameterVector& body_parameters);

            const OutputVector& get_initial_body_arguments() const;
            OutputVector& get_initial_body_arguments();
            void set_initial_body_arguments(const OutputVector& initial_body_arguments);

            const OutputVector& get_body_arguments() const;
            OutputVector& get_body_arguments();
            void set_body_arguments(const OutputVector& body_arguments);

            const OutputVector& get_outputs() const;
            OutputVector& get_outputs();
            void set_outputs(const OutputVector& outputs);

            const std::vector<SliceInput>& get_slice_inputs() const;
            std::vector<SliceInput>& get_slice_inputs();
            void set_slice_inputs(const std::vector<SliceInput>& slice_inputs);

            const std::vector<SliceOutput>& get_slice_outputs() const;
            std::vector<SliceOutput>& get_slice_outputs();
            void set_slice_outputs(const std::vector<SliceOutput>& slice_outputs);

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

        private:
            ParameterVector m_body_parameters;
            OutputVector m_initial_body_arguments;
            OutputVector m_body_arguments;
            OutputVector m_outputs;
            std::vector<SliceInput> m_slice_inputs;
            std::vector<SliceOutput> m_slice_outputs;
        };
    }
}
