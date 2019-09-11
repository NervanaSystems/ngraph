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

#include "ngraph/axis_set.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/util/arithmetic_reduction.hpp"

namespace ngraph
{
    namespace op
    {
        namespace v1
        {
            class ReduceSum : public util::ArithmeticReduction
            {
            public:
                NGRAPH_API
                static const std::string type_name;
                const std::string& description() const override { return type_name; }
                /// \brief Constructs a summation operation.
                ReduceSum() = default;
                /// \brief Constructs a summation operation.
                ///
                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to true it holds axes that are used for reduction.
                ReduceSum(const Output<Node>& arg,
                          const AxisSet& reduction_axes,
                          bool keep_dims = false);
                /// \brief Constructs a summation operation.
                ///
                /// \param arg The tensor to be summed.
                /// \param reduction_axes The axis positions (0-based) to be eliminated.
                /// \param keep_dims If set to 1 it holds axes that are used for reduction.
                ReduceSum(const Output<Node>& arg,
                          const Output<Node>& reduction_axes,
                          bool keep_dims = false);

                void validate_and_infer_types() override;

                size_t get_version() const override { return 1; }
                /// \return If set to 1 it holds axes that are used for reduction.
                /// For each such axis, output dimension is equal to 1.
                bool get_keep_dims() const { return m_keep_dims; }
                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                /// \return The default value for Sum.
                virtual std::shared_ptr<Node> get_default_value() const override;

            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const NodeVector& deltas) override;

            private:
                bool m_keep_dims;
            };
        }
    }
}