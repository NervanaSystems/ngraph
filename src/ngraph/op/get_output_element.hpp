//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

namespace ngraph
{
    namespace op
    {
        NodeVector get_output_elements(const std::shared_ptr<Node>& mon);

        /// \brief Operation to get an output from a node.
        class GetOutputElement : public Node
        {
        public:
            /// \brief Constructs a get-tuple-element operation.
            ///
            /// \param arg The input tuple.
            /// \param n The index of the tuple element to get.
            GetOutputElement(const std::shared_ptr<Node>& arg, size_t n);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return The index of the tuple element to get.
            size_t get_n() const { return m_n; }
            virtual NodeVector get_arguments() const override;

        protected:
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;
            size_t m_n;
        };
    }

    inline std::shared_ptr<Node> get_output_element(const std::shared_ptr<Node> node, size_t i = 0)
    {
        return ((i == 0) && node->get_output_size() == 1)
                   ? node
                   : std::make_shared<op::GetOutputElement>(node, i);
    }
}
