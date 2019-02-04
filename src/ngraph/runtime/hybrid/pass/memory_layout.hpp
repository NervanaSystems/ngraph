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
    namespace runtime
    {
        namespace hybrid
        {
            namespace pass
            {
                class MemoryLayout;
            }
        }
    }
}

class ngraph::runtime::hybrid::pass::MemoryLayout : public ngraph::pass::FunctionPass
{
public:
    MemoryLayout(size_t alignment = 64);
    bool run_on_function(std::shared_ptr<ngraph::Function>) override;

private:
    size_t m_alignment;
};
