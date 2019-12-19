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
            class NGRAPH_API ReverseSequence : public Op
            {
            public:
                static constexpr NodeTypeInfo type_info{"ReverseSequence", 0};
                const NodeTypeInfo& get_type_info() const override { return type_info; }
                ReverseSequence() = default;
                /// \brief Constructs an arcsin operation.
                ///
                /// \param arg Node that produces the input tensor.
                ReverseSequence(const Output<Node>& arg,
                                const Output<Node>& seq_lengths,
                                int64_t batch_axis,
                                int64_t seq_axis);

                void validate_and_infer_types() override;

                virtual std::shared_ptr<Node>
                    copy_with_new_args(const NodeVector& new_args) const override;

                size_t get_batch_axis() const { return m_normalized_batch_axis; }
                int64_t get_origin_batch_axis() const { return m_batch_axis; }
                void set_batch_axis(int64_t batch_axis) { m_batch_axis = batch_axis; }
                size_t get_sequence_axis() const { return m_normalized_seq_axis; }
                int64_t get_origin_sequence_axis() const { return m_seq_axis; }
                void set_sequence_axis(int64_t sequence_axis) { m_seq_axis = sequence_axis; }
            protected:
                virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                               const OutputVector& deltas) override;

            private:
                int64_t m_batch_axis;
                int64_t m_seq_axis;
                size_t m_normalized_batch_axis;
                size_t m_normalized_seq_axis;
            };
        }
        using v0::ReverseSequence;
    }
}
