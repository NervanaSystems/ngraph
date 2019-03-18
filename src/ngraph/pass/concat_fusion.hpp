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

#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class ConcatElimination;
        class SelfConcatFusion;
    }
}

class ngraph::pass::ConcatElimination : public ngraph::pass::GraphRewrite
{
public:
    ConcatElimination()
        : GraphRewrite()
    {
        construct_concat_elimination();
    }

private:
    void construct_concat_elimination();
};

class ngraph::pass::SelfConcatFusion : public ngraph::pass::FunctionPass
{
public:
    virtual bool run_on_function(std::shared_ptr<ngraph::Function> function) override;

private:
    void update_concat_pattern_vectors(const std::shared_ptr<Node>&, size_t);
    void remove_single_concat_op_pattern();

    std::vector<std::pair<NodeVector, std::vector<size_t>>> m_concat_pattern_vectors;
};
