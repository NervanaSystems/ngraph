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
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/util/fused_op.hpp"

namespace ngraph
{
    namespace op
    {
        /// \brief Splits the input tensor into a list of smaller tensors ("pieces")
        class Split : public ngraph::op::util::FusedOp
        {
        public:
            NGRAPH_API
            static const std::string type_name;
            const std::string& description() const override { return type_name; }
            /// \brief Constructs a Split op that evenly divides the input tensor.
            ///
            /// \param data - Node producing the input tensor
            /// \param axis - indicates an axis along which the input tensor should be split. Negative values mean counting from the back of the input tensor's shape.
            /// \param num_split - a number of "pieces" the input tensor will be split to
            Split(const std::shared_ptr<ngraph::Node>& data,
                  const int axis,
                  const size_t num_split);

            /// \brief Constructs a Split op that splits the input tensor into variable length "pieces"
            ///
            /// \param data - Node producing the input tensor
            /// \param axis - indicates an axis along which the input tensor should be split. Negative values mean counting from the back of the input tensor's shape.
            /// \param splits - a list of lengths that the input tensor should be split to. Use this constructor to split the input tensor to variable length chunks.
            Split(const std::shared_ptr<ngraph::Node>& data,
                  const int axis,
                  const std::vector<size_t>& splits);

            void pre_validate_and_infer_types() override;

            virtual NodeVector decompose_op() const override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_axis() const { return m_axis; }
            const std::vector<size_t>& get_splits() const { return m_splits; }
        private:
            /// \brief Adjusts the axis for negative values
            ///
            /// \note Negative values mean that the API consumer wants to point the axis location
            ///       from the back of the tensor. This is similar to the way NumPy works.
            ///
            /// \param axis - original axis value; negative values are accepted
            /// \param input_tensor_rank - rank of the input data tensor
            /// \return Returns a sum of parameters for negative axis value, or axis itself otherwise
            size_t adjust_axis_value(const int axis, const size_t input_tensor_rank) const;

            /// used internally for validation purposes, indicates which constructor was used
            bool m_split_evenly;
            int m_axis;
            size_t m_num_split;
            /// contains lengths of chunks that the input tensor will be split into
            std::vector<size_t> m_splits;
        };
    }
}
