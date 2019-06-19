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
        /// \brief SpaceToDepth permutes input tensor blocks of spatial data into depth dimension.
        ///
        /// \note  Values from the height and width dimensions are moved to the depth dimension.
        ///
        ///        Output node produces a tensor with shape:
        ///        [N, C * blocksize * blocksize, H / blocksize, W / blocksize]
        class SpaceToDepth : public ngraph::op::util::FusedOp
        {
        public:
            /// \brief Constructs a SpaceToDepth operation.
            ///
            /// \param data - Node producing the input tensor
            /// \param block_size - the size of the block of values to be moved
            SpaceToDepth(const std::shared_ptr<ngraph::Node>& data, std::size_t block_size);

            std::size_t get_block_size() const { return m_blocksize; }
            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            std::size_t m_blocksize;
        };
    }
}
