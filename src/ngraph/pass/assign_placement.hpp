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

#include <exception>
#include <functional>
#include <sstream>

#include "ngraph/pass/pass.hpp"
#include "ngraph/placement.hpp"

namespace ngraph
{
    namespace pass
    {
        class AssignPlacement : public NodePass
        {
        public:
            // TODO: make policy a class
            AssignPlacement(std::function<Placement(std::shared_ptr<Node>)> placement_policy);

        private:
            bool run_on_node(std::shared_ptr<Node> node) override;
            std::function<Placement(std::shared_ptr<Node>)> m_placement_policy;
        };
    }
}
