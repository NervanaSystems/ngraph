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

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class CommonSubexpressionElimination;
    }
}

class NGRAPH_API ngraph::pass::CommonSubexpressionElimination : public FunctionPass
{
public:
    CommonSubexpressionElimination()
        : FunctionPass()
    {
        set_property(PassProperty::REQUIRE_STATIC_SHAPE, true);
    }

    CommonSubexpressionElimination(
        const std::unordered_map<std::type_index,
                                 std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>&
            backend_cse_handlers)
        : FunctionPass()
        , m_backend_cse_handlers(backend_cse_handlers)
    {
        set_property(PassProperty::REQUIRE_STATIC_SHAPE, true);
    }

    std::unordered_map<std::type_index,
                       std::function<bool(std::shared_ptr<Node>, std::shared_ptr<Node>)>>
        m_backend_cse_handlers;

    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f);
};
