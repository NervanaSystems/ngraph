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
        /// \brief Dot product for a batch of Rank 2 tensors.
        class BatchDot : public Op
        {
        public:
            /// \brief Constructs a batch dot product operation.
            ///
            /// \param arg0 The node producing the first argument.
            /// \param arg1 The node producing the second argument.
            /// \param transpose_0 Apply transpose to arg0.
            /// \param transpose_1 Apply transpose to arg1.
            BatchDot(const std::shared_ptr<Node>& arg0,
                     const std::shared_ptr<Node>& arg1,
                     bool transpose_0 = false,
                     bool transpose_1 = false);

            bool get_transpose_arg0() const { return m_transpose_arg0; }
            bool get_transpose_arg1() const { return m_transpose_arg1; }
            virtual void validate_and_infer_types() override;

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            bool m_transpose_arg0;
            bool m_transpose_arg1;
        };
    }
}
