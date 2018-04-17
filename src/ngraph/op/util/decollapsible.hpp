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

#include "ngraph/node_vector.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/requires_tensor_view_args.hpp"

namespace ngraph
{
    namespace op
    {
        namespace util
        {
            /// \brief Abstract base class for ops on tensors views.
            class Decollapsible : public RequiresTensorViewArgs
            {
            public:
                std::shared_ptr<Node> get_original_node() { return m_original_node; }
            protected:
                /// \brief Constructs an operation on tensor view arguments.
                ///
                /// \param args The nodes producing this node's input tensors.
                Decollapsible(const std::string& node_type,
                              const NodeVector& args,
                              std::shared_ptr<Node> original_node);

            private:
                std::shared_ptr<Node> m_original_node;
            };
        }
    }
}
