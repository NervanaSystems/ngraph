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
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Performs a clipping operation on all elements of the input node
        ///
        /// All input values that are outside of the <min;max> range are set to 'min' or 'max'
        /// depending on which side of the <min;max> range they are. The values that fall into
        /// this range remain unchanged.
        class Clamp : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a Clamp node.
            ///
            /// \param data - Node producing the input tensor
            /// \param min - the lower bound of the <min;max> range
            /// \param max - the upper bound of the <min;max> range
            Clamp(const Output<ngraph::Node>& data, const double min, const double max);

            void pre_validate_and_infer_types() override;

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            double get_min() const { return m_min; }
            double get_max() const { return m_max; }
        private:
            const double m_min;
            const double m_max;
        };
    }
}
