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

#include "ngraph/op/op.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v0
        {
            // clang-format off
        /// \brief Elementwise Local Response Normalization (LRN) operation.
        ///
        /// ## Inputs
        ///
        /// |       | Type                                    | Description                                     |
        /// | ----- | --------------------------------------- | ----------------------------------------------- |
        /// | `arg` | \f$N[n, c, d_1,\dots,d_n]~(n \geq 0)\f$ | A tensor of any shape and numeric element type. |
        ///
        /// ## Output
        ///
        /// | Type                         | Description                                                                                                                                                                                  |
        /// | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
        /// | \f$N[n, c, d_1,\dots,d_n]\f$ | The tensor \f$T\f$, where \f$T[n, c, d_1,\dots,d_n] = \frac{N[n,i,d_1,\dots,d_n]}{ (bias + alpha * (\sum_{i=max(0,(nsize-1)/2)}^{min(C, (nsize-1)/2)+1} N[n,i,d_1,\dots,d_n]^{2}) ^ {2})}\f$ |
            // clang-format on
            class NGRAPH_API LRN : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"LRN", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                /// \brief Constructs a LRN operation.
                LRN() = default;
                /// \brief Constructs a LRN operation.
                ///
                /// \param arg Node that produces the input tensor.
                LRN(const Output<Node>& arg, double alpha, double beta, double bias, size_t size);

                LRN(const Output<Node>& arg,
                    const Output<Node>& axes,
                    double alpha,
                    double beta,
                    double bias,
                    size_t size);

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;
                void validate_and_infer_types() override;

                double get_alpha() const { return m_alpha; }
                void set_alpha(double alpha) { m_alpha = alpha; }
                double get_beta() const { return m_beta; }
                void set_beta(double beta) { m_beta = beta; }
                double get_bias() const { return m_bias; }
                void set_bias(double bias) { m_bias = bias; }
                size_t get_nsize() const { return m_size; }
                void set_nsize(size_t size) { m_size = size; }
                AxisSet get_reduction_axes() const;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

                double m_alpha;
                double m_beta;
                double m_bias;
                size_t m_size;
            };
        }
        using v0::LRN;
    }
}
