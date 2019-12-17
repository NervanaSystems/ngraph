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
        namespace v0
        {
            /// \brief  Iterate a body over tensors, accumulating into tensors.
            class NGRAPH_API TensorIterator : public util::FusedOp
            {
            public:
                static constexpr NodeTypeInfo type_info{"TensorIterator", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                // Forward declarations
                class SliceInputDescription;
                class MergedInputDescription;
                class InvariantInputDescription;

                TensorIterator() = default;
                TensorIterator(const OutputVector& values);

                class NGRAPH_API BodyLambda : public Lambda
                {
                public:
                    static constexpr DiscreteTypeInfo type_info{"BodyLamdba", 0};
                    const DiscreteTypeInfo& get_type_info() const { return type_info; }
                    BodyLambda(const OutputVector& outputs, const ParameterVector& parameters)
                        : Lambda(outputs, parameters)
                    {
                    }
                    BodyLambda(const ResultVector& results, const ParameterVector& parameters)
                        : Lambda(results, parameters)
                    {
                    }
                };

                /// \brief Describes a connection between a TensorIterator input and the body.
                class InputDescription
                {
                protected:
                    /// \param input_index Position of the TensorIterator input
                    /// \param body_parameter Body parameter to receive input
                    InputDescription(uint64_t input_index, uint64_t body_parameter_index);

                public:
                    virtual ~InputDescription() {}
                    virtual std::shared_ptr<InputDescription> copy() const = 0;

                    virtual const DiscreteTypeInfo& get_type_info() const = 0;

                    uint64_t m_input_index;
                    uint64_t m_body_parameter_index;
                };

                /// \brief Describes a body input formed from slices of an input to
                /// TensorIterator.
                class NGRAPH_API SliceInputDescription : public InputDescription
                {
                public:
                    static constexpr DiscreteTypeInfo type_info{"SliceInputDescription", 0};
                    const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                    /// \param input_index Position of the TensorIterator input
                    /// \param body_parameter_index Body parameter position to receive input
                    /// \param start First index for slices
                    /// \param stride Step amount for slices
                    /// \param part_size Width of slices
                    /// \param end Last index for slices
                    /// \param axis Axis being sliced
                    SliceInputDescription(uint64_t input_index,
                                          uint64_t body_parameter_index,
                                          int64_t start,
                                          int64_t stride,
                                          int64_t part_size,
                                          int64_t end,
                                          int64_t axis);
                    std::shared_ptr<InputDescription> copy() const override;

                    int64_t m_start;
                    int64_t m_stride;
                    int64_t m_part_size;
                    int64_t m_end;
                    int64_t m_axis;
                };

                /// \brief Describes a body input initialized from a TensorIterator input on the
                /// first
                /// iteration, and then a body output thereafter.
                class NGRAPH_API MergedInputDescription : public InputDescription
                {
                public:
                    static constexpr DiscreteTypeInfo type_info{"MergedInputDescription", 0};
                    const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                    /// \param input_index Position of the TensorIterator input supplying a
                    /// value to
                    /// body_parameter
                    /// for the initial iteration.
                    /// \param body_parameter_index Body parameter position to receive input.
                    /// \param body_value_index Body value to supply body_parameter for
                    /// successive
                    /// iterations.
                    MergedInputDescription(uint64_t input_index,
                                           uint64_t body_parameter_index,
                                           uint64_t body_value_index);
                    std::shared_ptr<InputDescription> copy() const override;

                    uint64_t m_body_value_index;
                };

                class NGRAPH_API InvariantInputDescription : public InputDescription
                {
                public:
                    static constexpr DiscreteTypeInfo type_info{"InvariantInputDescription", 0};
                    const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                    InvariantInputDescription(uint64_t input_index, uint64_t body_parameter_index);
                    std::shared_ptr<InputDescription> copy() const override;
                };

                // Forward declarations
                class ConcatOutputDescription;
                class BodyOutputDescription;

                /// \brief Describes how a TensorIterator output is produced from the body.
                class OutputDescription
                {
                protected:
                    /// \param body_value_index A body value that produces the output
                    /// \param output_index The TensorIterator output index
                    OutputDescription(uint64_t body_value_index, uint64_t output_index);

                public:
                    virtual ~OutputDescription() {}
                    virtual std::shared_ptr<OutputDescription> copy() const = 0;
                    virtual const DiscreteTypeInfo& get_type_info() const = 0;

                    uint64_t m_body_value_index;
                    uint64_t m_output_index;
                };

                /// \brief Produces an output by concatenating an output from each iteration
                class NGRAPH_API ConcatOutputDescription : public OutputDescription
                {
                public:
                    static constexpr DiscreteTypeInfo type_info{"ConcatOutputDescription", 0};
                    const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                    /// \param body_value_index A body value that produces the output
                    /// \param output_index The TensorIterator output index
                    /// \param start First index for slices
                    /// \param stride Step amount for slices
                    /// \param part_size Width of slices
                    /// \param end Last index for slices
                    /// \param axis Axis being sliced
                    ConcatOutputDescription(uint64_t body_value_index,
                                            uint64_t output_index,
                                            int64_t start,
                                            int64_t stride,
                                            int64_t part_size,
                                            int64_t end,
                                            int64_t axis);

                    virtual std::shared_ptr<OutputDescription> copy() const override;

                    int64_t m_start;
                    int64_t m_stride;
                    int64_t m_part_size;
                    int64_t m_end;
                    int64_t m_axis;
                };

                /// \brief Produces an output from a specific iteration
                class NGRAPH_API BodyOutputDescription : public OutputDescription
                {
                public:
                    static constexpr DiscreteTypeInfo type_info{"BodyOutputDescription", 0};
                    const DiscreteTypeInfo& get_type_info() const override { return type_info; }
                    /// \param body_value_index A body value that produces the output
                    /// \param output_index The TensorIterator output index
                    /// \param iteration which iteration (typically -1, final) will supply the
                    /// value
                    BodyOutputDescription(uint64_t body_value_index,
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
                /// \brief Indicates that a body parameter has an initial value in the first
                /// iteration
                /// and computed value thereafter
                /// \param initial_value Value for the parameter in first iteration. This will
                /// be added
                /// as an input to TensorIterator.
                /// \param successive_value Value for the parameter in successive iterations.
                /// The
                /// value is what is active in the most recent completed iteration.
                void set_merged_input(const std::shared_ptr<Parameter>& body_parameter,
                                      const Output<Node>& initial_value,
                                      const Output<Node>& successive_value);
                /// \brief Indicates that a body parameter has an invariant value during
                /// iteration that
                /// may depend on values computed outside of the iteration
                /// \param body_parameter The body parameter
                /// \param value The value supplied as an input to the block
                void set_invariant_input(const std::shared_ptr<Parameter>& body_parameter,
                                         const Output<Node>& value);
                /// \brief Gets a value for a particular iteration point
                /// \param body_value The value
                /// \param iteration The iteration that supplies the value. Negative values are
                /// from the
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
                /// \return the body of the iteration
                std::shared_ptr<BodyLambda> get_body() const { return m_body; }
                /// \param body set the body of the iteration
                void set_body(const std::shared_ptr<BodyLambda>& body) { m_body = body; }
                /// \return a reference to the input descriptions.
                const std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions() const
                {
                    return m_input_descriptions;
                }
                /// \return a reference to the input descriptions. Can add input descriptions
                /// before
                /// validation.
                std::vector<std::shared_ptr<InputDescription>>& get_input_descriptions()
                {
                    return m_input_descriptions;
                }

                /// \return a reference to the output descriptions.
                const std::vector<std::shared_ptr<OutputDescription>>&
                    get_output_descriptions() const
                {
                    return m_output_descriptions;
                }

                /// \return a reference to the output descriptions. Can add output descriptions
                /// before
                /// validation.
                std::vector<std::shared_ptr<OutputDescription>>& get_output_descriptions()
                {
                    return m_output_descriptions;
                }

                virtual void validate_and_infer_types() override;
                void revalidate_and_infer_types_for_body_ops();

                int64_t get_num_iterations() const { return m_num_iterations; }
                void set_num_iterations(int64_t num_iterations)
                {
                    m_num_iterations = num_iterations;
                }

            private:
                // Find an input corresponding to value, adding one if necessary.
                Input<Node> input_for_value(const Output<Node>& value);

                std::shared_ptr<BodyLambda> m_body;
                std::vector<std::shared_ptr<InputDescription>> m_input_descriptions;
                std::vector<std::shared_ptr<OutputDescription>> m_output_descriptions;

                int64_t m_num_iterations = -1;
            };
        }
        using v0::TensorIterator;
    }
}
