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

#include "ngraph/op/util/requires_tensor_view_args.hpp"
#include "ngraph/util.hpp"

namespace ngraph
{
    namespace op
    {
        class Sigmoid : public util::RequiresTensorViewArgs
        {
        public:
            Sigmoid(std::shared_ptr<Node> input);
            Shape get_input_shape() const { return m_shape_input; }
            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
            virtual void generate_adjoints(autodiff::Adjoints& adjoints,
                                           const NodeVector& deltas) override;

        private:
            Shape m_shape_input;
        };

        /// \brief Elementwise SigmoidBackprop operation.
        ///
        class SigmoidBackprop : public util::RequiresTensorViewArgs
        {
        public:
            /// \brief Constructs a SigmoidBackprop operation.
            ///
            /// \param arg Node that produces the Sigmoid forward input tensor.
            SigmoidBackprop(std::shared_ptr<ngraph::Node> arg, std::shared_ptr<ngraph::Node> delta);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;
        };
    }
}
