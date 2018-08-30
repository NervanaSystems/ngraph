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

#include <set>
#include <sstream>
#include <string>

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class VisualizeTree;
    }
}

class ngraph::pass::VisualizeTree : public ModulePass
{
public:
    VisualizeTree(const std::string& file_name);
    bool run_on_module(std::vector<std::shared_ptr<ngraph::Function>>&) override;

    static std::string get_file_ext();

private:
    std::string add_attributes(std::shared_ptr<Node> node);
    std::string get_attributes(std::shared_ptr<Node> node);
    void render() const;

    std::stringstream m_ss;
    std::string m_name;
    std::set<std::shared_ptr<Node>> m_nodes_with_attributes;
};
