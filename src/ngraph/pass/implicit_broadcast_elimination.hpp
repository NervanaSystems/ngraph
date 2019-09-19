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

#include "ngraph/op/util/broadcasting.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        NodeVector explicit_broadcast(std::shared_ptr<Node>& node)
        {
            NodeVector rc;
            if (node->supports_auto_broadcast())
            {
                if (node->get_autob().m_type == op::AutoBroadcastType::NONE)
                {
                    rc = node->get_arguments();
                }
                else if (node->get_autob().m_type == op::AutoBroadcastType::NUMPY)
                {
                    rc = op::numpy_style_broadcast(node->get_arguments());
                }
                else
                {
                    throw ngraph_error("Unsupported implicit broadcast type");
                }
            }
            return rc;
        }

        class ImplicitBroadcastElimination : public NodePass
        {
        public:
            bool run_on_node(std::shared_ptr<ngraph::Node> node) override;
        };
    }
}
