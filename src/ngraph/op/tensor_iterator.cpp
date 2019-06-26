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

#include "ngraph/op/tensor_iterator.hpp"

using namespace std;
using namespace ngraph;

const string op::TensorIterator::type_name{"TensorIterator"};

op::TensorIterator::TensorIterator(const OutputVector& body_inputs,
                                   const ParameterVector& body_parameters,
                                   const OutputVector& body_outputs,
                                   const OutputVector& tensor_iterator_outputs,
                                   const AxisSet& sequence_inputs,
                                   const AxisSet& sequence_outputs)
    : Op(body_inputs)
    , m_body_parameters(body_parameters)
    , m_body_outputs(body_outputs)
    , m_tensor_iterator_outputs(tensor_iterator_outputs)
    , m_sequence_inputs(sequence_inputs)
    , m_sequence_outputs(sequence_outputs)
{
    constructor_validate_and_infer_types();
}

void op::TensorIterator::validate_and_infer_types()
{
    // The number of iterations is determined by the shortest sequence input
    size_t iteration_count{0};
    // If true, iteration count is dynamic
    bool iteration_count_dynamic{false};
    // true when we something about the count
    bool iteration_count_valid{false};
    // Shapes of the inputs that would be passed to iterator. 
    // For sequences, the sequence axis is removed, for everything
    // else, the shape is unchanged.
    vector<PartialShape> iterator_shapes;
    for (size_t axis = 0; axis < get_input_size(); ++axis)
    {
        PartialShape sequence_shape = get_input_partial_shape(axis);
        PartialShape iterator_shape = sequence_shape;
        Rank sequence_rank = sequence_shape.rank();

        if (m_sequence_inputs.find(axis) != m_sequence_inputs.end())
        {
            if (sequence_rank.is_dynamic())
            {
                // Can't determine the sequence length
                iteration_count_dynamic = true;
            }
            else
            {
                NODE_VALIDATION_CHECK(this,
                                      static_cast<size_t>(sequence_shape.rank()) != 0,
                                      "Input ",
                                      axis,
                                      " is specified to be a sequence but is scalar.");
                Dimension sequence_dim = sequence_shape[0];
                vector<Dimension> dimensions = static_cast<vector<Dimension>>(sequence_shape);
                dimensions.erase(dimensions.begin());
                iterator_shape = PartialShape(dimensions);

                if (sequence_dim.is_dynamic())
                {
                    // Can't determine the sequence length
                    iteration_count_dynamic = true;
                }
                else
                {
                    size_t sequence_length = static_cast<size_t>(sequence_dim);
                    if (!iteration_count_valid || (sequence_length < iteration_count))
                    {
                        iteration_count = sequence_length;
                        iteration_count_valid = true;
                    }
                }
            }
        }
        iterator_shapes.push_back(iterator_shape);
    }
    // Now we make sure the parameters match.
}

const ParameterVector& op::TensorIterator::get_body_parameters() const
{
    return m_body_parameters;
}

ParameterVector& op::TensorIterator::get_body_parameters()
{
    return m_body_parameters;
}

void op::TensorIterator::set_body_parameters(const ParameterVector& body_parameters)
{
    m_body_parameters = body_parameters;
}

const OutputVector& op::TensorIterator::get_body_outputs() const
{
    return m_body_outputs;
}

OutputVector& op::TensorIterator::get_body_outputs()
{
    return m_body_outputs;
}

void op::TensorIterator::set_body_outputs(const OutputVector& body_outputs)
{
    m_body_outputs = body_outputs;
}

const OutputVector& op::TensorIterator::get_tensor_iterator_outputs() const
{
    return m_tensor_iterator_outputs;
}

OutputVector& op::TensorIterator::get_tensor_iterator_outputs()
{
    return m_tensor_iterator_outputs;
}

void op::TensorIterator::set_tensor_iterator_outputs(const OutputVector& tensor_iterator_outputs)
{
    m_tensor_iterator_outputs = tensor_iterator_outputs;
}

const AxisSet& op::TensorIterator::get_sequence_inputs() const
{
    return m_sequence_inputs;
}

AxisSet& op::TensorIterator::get_sequence_inputs()
{
    return m_sequence_inputs;
}

void op::TensorIterator::set_sequence_inputs(const AxisSet& sequence_inputs)
{
    m_sequence_inputs = sequence_inputs;
}

const AxisSet& op::TensorIterator::get_sequence_outputs() const
{
    return m_sequence_outputs;
}

AxisSet& op::TensorIterator::get_sequence_outputs()
{
    return m_sequence_outputs;
}

void op::TensorIterator::set_sequence_outputs(const AxisSet& sequence_outputs)
{
    m_sequence_outputs = sequence_outputs;
}
std::shared_ptr<Node> op::TensorIterator::copy_with_new_args(const NodeVector& new_args) const
{
    OutputVector output_vector;
    for (auto arg : new_args)
    {
        output_vector.push_back(arg);
    }
    return make_shared<TensorIterator>(output_vector,
                                       m_body_parameters,
                                       m_body_outputs,
                                       m_tensor_iterator_outputs,
                                       m_sequence_inputs,
                                       m_sequence_outputs);
}
