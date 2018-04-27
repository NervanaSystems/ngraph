/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include <memory>

#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        class ReverseSequence : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs an arcsin operation.
            ///
            /// \param arg Node that produces the input tensor.
            ReverseSequence(const std::shared_ptr<Node> arg,
                            size_t batch_axis,
                            size_t seq_axis,
                            const Shape& seq_lengths);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            size_t get_batch_axis() { return m_batch_axis; }
            size_t get_sequence_axis() { return m_seq_axis; }
            Shape get_sequence_lengths() { return m_seq_lengths; }
        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            bool m_batch_axis{0};
            bool m_seq_axis{0};
            Shape m_seq_lengths;
        };
    }
}
