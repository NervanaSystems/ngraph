//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include "ngraph/op/util/unary_elementwise_arithmetic.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Elementwise Local Response Normalization (LRN) operation.
        ///
        /// ## Inputs
        ///
        /// |       | Type                              | Description                                     |
        /// | ----- | --------------------------------- | ----------------------------------------------- |
        /// | `arg` | \f$N[n, c, d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                   | Description                                                                          |
        /// | ---------------------- | ------------------------------------------------------------------------------------ |
        /// | \f$N[n, c, d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[n, c, d_1,\dots,d_n] = \frac{N[n,i,d_1,\dots,d_n]}{ (bias + alpha * (\sum_{i=max(0,(nsize-1)/2)}^{min(C, (nsize-1)/2)+1} N[n,i,d_1,\dots,d_n]^{2}) ^ {2})}\f$ |
        class LRN : public util::UnaryElementwiseArithmetic
        {
        public:
            /// \brief Constructs a LRN operation.
            ///
            /// \param arg Node that produces the input tensor.
            LRN(const std::shared_ptr<Node>& arg,
                double alpha,
                double beta,
                double bias,
                size_t size);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            double get_alpha() const { return m_alpha; }
            double get_beta() const { return m_beta; }
            double get_bias() const { return m_bias; }
            size_t get_nsize() const { return m_size; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

            double m_alpha;
            double m_beta;
            double m_bias;
            size_t m_size;
        };
    }
}
