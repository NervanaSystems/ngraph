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

#include <unordered_map>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class CommonFunctionCollection;
    }
}

class ngraph::pass::CommonFunctionCollection : public ModulePass
{
public:
    CommonFunctionCollection(std::function<std::string(Node&, std::string)> function_emitter,
                             std::unordered_map<Node*, Node*>& result_map);
    virtual ~CommonFunctionCollection();

    bool run_on_module(std::vector<std::shared_ptr<ngraph::Function>>&) override;

    static void emit_function(codegen::CodeWriter&,
                              const std::unordered_map<Node*, Node*>& node_function_map,
                              std::function<std::string(Node&, std::string)> emitter);

private:
    std::function<std::string(Node&, std::string)> m_emit_op_as_function;
    std::unordered_map<Node*, Node*>& m_node_function_map;
};
