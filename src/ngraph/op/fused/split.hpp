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

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Splits the input tensor into a list of smaller, evenly-sized tensors
        class Split : public ngraph::op::util::FusedOp
        {
        public:
            /// \brief Constructs a Clamp node.
            ///
            /// \param data - Node producing the input tensor
            /// \param axis - indicates an axis along which the input tensor should be split. Negative values mean counting from the back of the input tensor's shape.
            /// \param num_split - a number of "pieces" the input tensor will be split to
            Split(const std::shared_ptr<ngraph::Node>& data, const int axis, const size_t num_split);

            void pre_validate_and_infer_types() override;

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_axis() const { return m_axis; }
            size_t get_num_split() const { return m_num_split; }
        private:
            size_t m_axis;
            const size_t m_num_split;
        };
    }
}
