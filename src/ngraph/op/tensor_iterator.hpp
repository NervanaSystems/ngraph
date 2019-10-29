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

#include <vector>

#include "ngraph/lambda.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief  Iterate a body over tensors, accumulating into tensors.
        class TensorIterator : public util::FusedOp
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"TensorIterator", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            // Forward declarations
            class SliceInputDescription;
            class BodyConnectionInputDescription;
            class ConstantInputDescription;

            TensorIterator() = default;
            TensorIterator(const OutputVector& values);

            class BodyLambda : public Lambda
            {
            public:
                static constexpr DiscreteTypeInfo type_info{"BodyLamdba", 0};
                const DiscreteTypeInfo& get_type_info() const { return type_info; }
                BodyLambda(const OutputVector& outputs, const ParameterVector& parameters)
                    : Lambda(outputs, parameters)
                {
                }
            };

            /// \brief Describes a connection between a TensorIterator input and the body.
            class InputDescription
            {
            protected:
                /// \param input_index Position of the TensorIterator input
                /// \param body_parameter Body parameter to receive input
                InputDescription(uint64_t input_index,
                                 const std::shared_ptr<Parameter>& body_parameter);

            public:
                virtual ~InputDescription() {}
                virtual std::shared_ptr<InputDescription> copy() const = 0;

                virtual const DiscreteTypeInfo& get_type_info() const = 0;

                uint64_t m_input_index;
                std::shared_ptr<Parameter> m_body_parameter;
            };

            /// \brief Describes a body input formed from slices of an input to TensorIterator.
            class SliceInputDescription : public InputDescription
            {
            public:
                static constexpr DiscreteTypeInfo type_info{"SliceInputDescription", 0};
                const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                /// \param input_index Position of the TensorIterator input
                /// \param body_parameter Body parameter to receive input
                /// \param start First index for slices
                /// \param stride Step amount for slices
                /// \param part_size Width of slices
                /// \param end Last index for slices
                /// \param axis Axis being sliced
                SliceInputDescription(uint64_t input_index,
                                      const std::shared_ptr<Parameter>& body_parameter,
                                      int64_t start,
                                      int64_t stride,
                                      uint64_t part_size,
                                      int64_t end,
                                      int64_t axis);
                std::shared_ptr<InputDescription> copy() const override;

                int64_t m_start;
                int64_t m_stride;
                uint64_t m_part_size;
                int64_t m_end;
                int64_t m_axis;
            };

            /// \brief Describes a body input initialized from a TensorIterator input on the first
            /// iteration, and then a body output thereafter.
            class BodyConnectionInputDescription : public InputDescription
            {
            public:
                static constexpr DiscreteTypeInfo type_info{"BodyConnectionInputDescription", 0};
                const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                /// \param input_index Position of the TensorIterator input supplying a value to
                /// body_parameter
                /// for the initial iteration.
                /// \param body_parameter Body parameter to receive input.
                /// \param body_value Body value to supply body_parameter for successive iterations.
                BodyConnectionInputDescription(uint64_t input_index,
                                               const std::shared_ptr<Parameter>& body_parameter,
                                               const Output<Node>& body_value);
                std::shared_ptr<InputDescription> copy() const override;

                Output<Node> m_body_value;
            };

            class ConstantInputDescription : public InputDescription
            {
            public:
                static constexpr DiscreteTypeInfo type_info{"ConstantInputDescription", 0};
                const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                ConstantInputDescription(uint64_t input_index,
                                         const std::shared_ptr<op::Parameter>& body_parameter);
                std::shared_ptr<InputDescription> copy() const override;
            };

            // Forward declarations
            class ConcatOutputDescription;
            class BodyOutputDescription;

            /// \brief Describes how a TensorIterator output is produced from the body.
            class OutputDescription
            {
            protected:
                /// \param body_value A body value that produces the output
                /// \param output_index The TensorIterator output index
                OutputDescription(const Output<Node>& body_value, uint64_t output_index);

            public:
                virtual ~OutputDescription() {}
                virtual std::shared_ptr<OutputDescription> copy() const = 0;
                virtual const DiscreteTypeInfo& get_type_info() const = 0;

                Output<Node> m_body_value;
                uint64_t m_output_index;
            };

            /// \brief Produces an output by concatenating an output from each iteration
            class ConcatOutputDescription : public OutputDescription
            {
            public:
                static constexpr DiscreteTypeInfo type_info{"ConcatOutputDescription", 0};
                const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                /// \param body_value A body value that produces the output
                /// \param output_index The TensorIterator output index
                /// \param start First index for slices
                /// \param stride Step amount for slices
                /// \param part_size Width of slices
                /// \param end Last index for slices
                /// \param axis Axis being sliced
                ConcatOutputDescription(const Output<Node>& body_value,
                                        uint64_t output_index,
                                        int64_t start,
                                        int64_t stride,
                                        uint64_t part_size,
                                        int64_t end,
                                        int64_t axis);

                virtual std::shared_ptr<OutputDescription> copy() const override;

                int64_t m_start;
                int64_t m_stride;
                uint64_t m_part_size;
                int64_t m_end;
                int64_t m_axis;
            };

            /// \brief Produces an output from a specific iteration
            class BodyOutputDescription : public OutputDescription
            {
            public:
                static constexpr DiscreteTypeInfo type_info{"BodyOutputDescription", 0};
                const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                /// \param body_value A body value that produces the output
                /// \param output_index The TensorIterator output index
                /// \param iteration which iteration (typically -1, final) will supply the value
                BodyOutputDescription(const Output<Node>& body_value,
                                      uint64_t output_index,
                                      int64_t iteration);
                std::shared_ptr<OutputDescription> copy() const override;

                int64_t m_iteration;
            };

            /// \brief Indicate that a body parameter comes from slices of a value
            /// \param parameter The parameter to receive the slices
            /// \param value The value to be sliced. This will be added as an input to
            /// TensorIterator.
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
            /// \param initial_value Value for the parameter in first iteration. This will be added
            /// as an input to TensorIterator.
            /// \param successive_value Value for the parameter in successive iterations. The
            /// value is what is active in the most recent completed iteration.
            void set_initialized_input(const std::shared_ptr<Parameter>& body_parameter,
                                       const Output<Node>& initial_value,
                                       const Output<Node>& successive_value);
            /// \brief Indicates that a body parameter has a constant value during iteration that
            /// may depend on values computed outside of the iteration
            /// \param body_parameter The body parameter
            /// \param value The value supplied as an input to the block
            void set_constant_input(const std::shared_ptr<Parameter>& body_parameter,
                                    const Output<Node>& value);
            /// \brief Gets a value for a particular iteration point
            /// \param body_value The value
            /// \param iteration The iteration that supplies the value. Negative values are from the
            /// last iteration.
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

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;
            NodeVector decompose_op() const override;

            const std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions() const
            {
                return m_input_descriptions;
            }

            std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions()
            {
                return m_input_descriptions;
            }

            const std::vector<std::shared_ptr<OutputDescription>>& get_output_descriptions() const
            {
                return m_output_descriptions;
            }

            std::vector<std::shared_ptr<OutputDescription>>& get_output_descriptions()
            {
                return m_output_descriptions;
            }

            virtual void validate_and_infer_types() override;

            int64_t get_num_iterations() const { return m_num_iterations; }
        private:
            // Find an input corresponding to value, adding one if necessary.
            Input<Node> input_for_value(const Output<Node>& value);

            std::vector<std::shared_ptr<InputDescription>> m_input_descriptions;
            std::vector<std::shared_ptr<OutputDescription>> m_output_descriptions;

            int64_t m_num_iterations = -1;
        };
    }
}
