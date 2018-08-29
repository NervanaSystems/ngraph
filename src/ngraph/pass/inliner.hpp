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

#include <functional>
#include <memory>
#include <vector>
#include "ngraph/op/function_call.hpp"
#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace pass
    {
        class Inliner;
        class InliningHeuristics;
        class InlineSmallCalls;
    }
}

class ngraph::pass::InliningHeuristics
{
public:
    virtual std::vector<std::shared_ptr<ngraph::op::FunctionCall>>
        create_inlining_plan(std::shared_ptr<ngraph::Function> f, size_t depth) = 0;
    virtual ~InliningHeuristics() {}
};

class ngraph::pass::InlineSmallCalls : public ngraph::pass::InliningHeuristics
{
public:
    InlineSmallCalls(size_t call_size_limit, size_t depth)
        : InliningHeuristics()
        , m_call_size_limit(call_size_limit)
        , m_depth(depth)
    {
    }

    std::vector<std::shared_ptr<ngraph::op::FunctionCall>>
        create_inlining_plan(std::shared_ptr<ngraph::Function> f, size_t depth) override;
    virtual ~InlineSmallCalls() override {}
private:
    size_t m_call_size_limit;
    size_t m_depth;
};

class ngraph::pass::Inliner : public ModulePass
{
public:
    Inliner(std::shared_ptr<InliningHeuristics> ih)
        : ModulePass()
        , m_inlining_heuristics(ih)
        , m_depth(0)
    {
    }

    static bool inline_function_call(std::shared_ptr<ngraph::Node> inlinee,
                                     std::shared_ptr<ngraph::Function> caller);

    bool run_on_function_call(std::shared_ptr<ngraph::op::FunctionCall> fc);
    void run_on_functions(std::vector<std::shared_ptr<ngraph::op::FunctionCall>>,
                          std::shared_ptr<ngraph::Function> caller);
    bool run_on_module(std::vector<std::shared_ptr<ngraph::Function>>&) override;

private:
    std::shared_ptr<InliningHeuristics> m_inlining_heuristics;
    size_t m_depth;
};
