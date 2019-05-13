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
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Permutes data in the channel dimension of the input
        class ShuffleChannels : public ngraph::op::util::FusedOp
        {
        public:
            /// \brief Constructs a ShuffleChannels node.
            ///
            /// \param data - Node producing the input tensor
            /// \param axis - channel dimension index in the data tensor
            /// \param groups - a number of groups the channel dimension specified by axis should be split to
            ShuffleChannels(const std::shared_ptr<ngraph::Node>& data, const size_t axis, const size_t groups);

            void pre_validate_and_infer_types() override;

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            const size_t get_axis() const { return m_axis; }
            const size_t get_groups() const { return m_groups; }
        private:
            const size_t m_axis = 1;
            const size_t m_groups = 1;
        };
    }
}
