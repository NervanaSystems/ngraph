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

#include <memory>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Concatenation operation.
        class Concat : public Op
        {
        public:
            NGRAPH_API
            static constexpr NodeTypeInfo type_info{"Concat", 0};
            const NodeTypeInfo& get_type_info() const override { return type_info; }
            /// \brief Constructs a concatenation operation.
            Concat() = default;
            /// \brief Constructs a concatenation operation.
            ///
            /// \param args               The outputs producing the input tensors.
            /// \param axis The axis along which to concatenate the input tensors.
            Concat(const OutputVector& args, size_t axis);

            /// \brief Constructs a concatenation operation.
            ///
            /// \param args               The nodes producing the input tensors.
            /// \param axis The axis along which to concatenate the input tensors.
            Concat(const NodeVector& args, size_t axis);

            void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The concatenation axis.
            size_t get_concatenation_axis() const { return get_axis(); }
            void set_concatenation_axis(size_t concatenation_axis) { set_axis(concatenation_axis); }
            /// \return The concatenation axis.
            size_t get_axis() const { return m_axis; }
            void set_axis(size_t axis) { m_axis = axis; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            size_t m_axis;
        };
    }
}
