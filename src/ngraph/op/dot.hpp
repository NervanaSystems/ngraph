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

#include <utility>

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Generalized dot product operation, including scalar-tensor product, matrix-vector product, and matrix multiplication.
        class Dot : public Op
        {
        public:
            /// \brief Constructs a dot product operation.
            ///
            /// \param arg0 The output producing the first argument.
            /// \param arg1 The output producing the second argument.
            /// \param reduction_axes_count The number of axes to dot.
            Dot(const NodeOutput& arg0,
                const NodeOutput& arg1,
                size_t reduction_axes_count,
                bool has_reduction_axes_count = true);

            /// \brief Constructs a dot product operation with default dot-axis selection depending on the inputs.
            ///
            /// If `arg0` or `arg1` is a scalar, there are no dot-axes. Else, there is one dot-axis.
            ///
            /// (Note that in particular, this results in scalar-tensor products where one or the other argument is
            /// a scalar, a matrix-vector products where `arg0` is a matrix and `arg1` is a vector, and a
            /// matrix multiplication where `arg0` and `arg1` are both matrices.)
            ///
            /// \param arg0 The output producing the first argument.
            /// \param arg1 The output producing the second argument.
            Dot(const NodeOutput& arg0, const NodeOutput& arg1);

            void validate_and_infer_types() override;

            size_t get_reduction_axes_count() const { return m_reduction_axes_count; }
            virtual std::shared_ptr<Node>
                copy_with_new_source_outputs(const OutputVector& new_source_outputs) const override
            {
                check_new_source_outputs_count(this, new_source_outputs);
                return std::make_shared<Dot>(
                    new_source_outputs.at(0), new_source_outputs.at(1), m_reduction_axes_count);
            }

        protected:
            size_t m_reduction_axes_count;
            bool m_has_reduction_axes_count;

            virtual void build_backprop(autodiff::Adjoints& adjoints,
                                        const OutputVector& deltas) override;
        };
    }
}
