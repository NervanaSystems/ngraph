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

#pragma once

#include "ngraph/node_vector.hpp"
#include "ngraph/pass/pass.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPUWorkspaceInsertion;
            }
        }
    }
}

class CPU_BACKEND_API ngraph::runtime::cpu::pass::CPUWorkspaceInsertion
    : public ngraph::pass::FunctionPass
{
public:
    CPUWorkspaceInsertion(ngraph::NodeVector& indices_list, bool return_indices = true)
        : FunctionPass()
        , m_indices_list(indices_list)
        , m_return_indices(return_indices)
    {
    }

    virtual bool run_on_function(std::shared_ptr<ngraph::Function> f);

private:
    ngraph::NodeVector& m_indices_list;
    bool m_return_indices;
    bool transform(ngraph::pattern::Matcher& m);
};
