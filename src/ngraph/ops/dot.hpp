// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <utility>

#include "ngraph/ops/op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Generalized dot product operation, including scalar-tensor product, matrix-vector product, and matrix multiplication.
        ///
        /// Takes two arguments `arg0` and `arg1`, with shapes \f$(i_1,\dots,i_n,j_1,\dots,j_m)\f$ and \f$(j_1,\dots,j_m,k_1,\dots,k_p)\f$ respectively,
        /// and produces an output tensor with shape \f$(i_1,\dots,i_n,k_1,\dots,k_p)\f$ by summing products along the \f$j\f$ dimensions.
        ///
        /// A few common cases are as follows:
        ///
        /// * If \f$m = 0\f$ and \f$n = 1\f$ or \f$p = 1\f$, the operation is a scalar-tensor product.
        /// * If \f$m = 1\f$, \f$n = 2\f$, and \f$p = 1\f$, the operation is a matrix-vector product.
        /// * If \f$m = 1\f$ and \f$n = p = 2\f$, the operation is a matrix multiplication.
        ///
        /// ## Parameters
        ///
        /// |                  | Description                                                                                      |
        /// | ---------------- | ------------------------------------------------------------------------------------------------ |
        /// | `dot_axis_count` | The number of axes to reduce through dot-product (corresponds to \f$m\f$ in the formulas above). |
        ///
        /// ## Inputs
        ///
        /// |        | Type                                                  | Description                                                                                                                                                                 |
        /// | ------ | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | `arg0` | \f$E[d_1,\dots,d_n,d'_1,\dots,d'_m]~(n,m \geq 0)\f$   | A tensor of any shape and element type.                                                                                                                                     |
        /// | `arg1` | \f$E[d'_1,\dots,d'_m,d''_1,\dots,d''_p]~(p \geq 0)\f$ | A tensor of any shape with the same element type as `arg0` and rank at least \f$m\f$, whose first \f$m\f$ dimensions match the last \f$m\f$ dimensions of `arg0`, in order. |
        ///
        /// ## Output
        ///
        /// | Type                                     | Description                                                                                                                                                                                                                                                                                                                                  |
        /// | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$E[d_1,\dots,d_n,d''_1,\dots,d''_p]\f$ | The tensor \f$T\f$, where \f$T[i_1,\dots,i_n,k_1,\dots,k_p] = \Sigma_{0 \le j_1 < d'_1, \dots, 0 \le j_m < d'_m}(\mathtt{arg0}[i_1,\dots,i_n,j_1,\dots,j_m] \cdot \mathtt{arg1}[j_1,\dots,j_m,k_1,\dots,k_p])\f$ or, if \f$m = 0\f$, \f$T[i_1,\dots,i_n,k_1,\dots,k_p] = \mathtt{arg0}[i_1,\dots,i_n] \cdot \mathtt{arg1}[k_1,\dots,k_p]\f$. |
        ///
        class Dot : public RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a dot product operation.
            ///
            /// \param arg0 The node producing the first argument.
            /// \param arg1 The node producing the second argument.
            /// \param dot_axis_count The number of axes to dot.
            Dot(const std::shared_ptr<Node>& arg0,
                const std::shared_ptr<Node>& arg1,
                size_t dot_axis_count);

            /// \brief Constructs a dot product operation with default dot-axis selection depending on the inputs.
            ///
            /// If `arg0` or `arg1` is a scalar, there are no dot-axes. Else, there is one dot-axis.
            ///
            /// (Note that in particular, this results in scalar-tensor products where one or the other argument is
            /// a scalar, a matrix-vector products where `arg0` is a matrix and `arg1` is a vector, and a
            /// matrix multiplication where `arg0` and `arg1` are both matrices.)
            ///
            /// \param arg0 The node producing the first argument.
            /// \param arg1 The node producing the second argument.
            Dot(const std::shared_ptr<Node>& arg0, const std::shared_ptr<Node>& arg1);

            size_t get_dot_axis_count() const { return m_dot_axis_count; }
            virtual std::shared_ptr<Node> copy_with_new_args(
                const std::vector<std::shared_ptr<Node>>& new_args) const override
            {
                if (new_args.size() != 2)
                    throw ngraph_error("Incorrect number of new arguments");
                return std::make_shared<Dot>(new_args.at(0), new_args.at(1), m_dot_axis_count);
            }

        protected:
            size_t m_dot_axis_count;

            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const std::shared_ptr<Node>& delta) override;
        };
    }
}
