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
        /// \brief  Iterate a body over tensors, accumulating into tensors
        class TensorIterator : public Op
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs an empty TensorIterator operation
            TensorIterator() = default;

            /// \param body_arguments sequences that will be iterated over.
            /// The shortest sequence determines the number of steps.
            /// \param body_parameters Parameters that will be bound to successive
            /// input sequence elements.
            /// \param body_outputs Values computed on each iteration. These will be sequences.
            /// body_outputs dependencies on parameters will be per-iteration, dependencies on
            /// nodes not dependent on parameters will be constant per iteration.
            /// \param outputs Values returned from TensorIterator.
            TensorIterator(const OutputVector& body_arguments,
                           const ParameterVector& body_parameters,
                           const OutputVector& body_outputs,
                           const OutputVector& outputs);

            void validate_and_infer_types() override;

            const ParameterVector& get_body_parameters() const;
            ParameterVector& get_body_parameters();
            void set_body_parameters(const ParameterVector& body_parameters);

            const OutputVector& get_body_outputs() const;
            OutputVector& get_body_outputs();
            void set_body_outputs(const OutputVector& body_outputs);

            const OutputVector& get_tensor_iterator_outputs() const;
            OutputVector& get_tensor_iterator_outputs();
            void set_tensor_iterator_outputs(const OutputVector& tensor_iterator_outputs);

            std::shared_ptr<Node> copy_with_new_args(const NodeVector& new_args) const override;

        private:
            ParameterVector m_body_parameters;
            OutputVector m_body_outputs;
            OutputVector m_tensor_iterator_outputs;
        };
    }
}
