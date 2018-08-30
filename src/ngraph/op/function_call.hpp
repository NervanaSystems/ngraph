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
        /// \brief %Function call operation.
        class FunctionCall : public Node
        {
        public:
            /// \brief Constructs a function call operation.
            ///
            /// \param function The function to be called.
            /// \param args The arguments for the function call.
            FunctionCall(std::shared_ptr<Function> function, const NodeVector& args);

            virtual std::shared_ptr<Node>
                copy_with_new_args(const NodeVector& new_args) const override;

            /// \return A singleton vector containing the function to be called.
            std::vector<std::shared_ptr<Function>> get_functions() const override;

        protected:
            std::shared_ptr<Function> m_function;
        };
    }
}
