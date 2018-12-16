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

#include <limits>
#include <list>
#include <sstream>

#include "ngraph/pass/pass.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPUMemoryAssignment;
            }
        }
    }
}
class ngraph::runtime::cpu::pass::CPUMemoryAssignment : public ngraph::pass::FunctionPass
{
public:
    CPUMemoryAssignment(size_t alignment = 1, bool disable_memory_sharing = false);
    bool run_on_function(std::shared_ptr<ngraph::Function>) override;

private:
    size_t m_alignment;
    bool m_disable_memory_sharing;
    std::set<descriptor::Tensor*> m_tensor_caching;
};
