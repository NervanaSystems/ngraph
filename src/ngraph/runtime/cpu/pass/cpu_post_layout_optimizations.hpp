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
#include "ngraph/runtime/cpu/cpu_backend_visibility.h"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace pass
            {
                class CPUPostLayoutOptimizations;
            }
        }
    }
}

class CPU_BACKEND_API ngraph::runtime::cpu::pass::CPUPostLayoutOptimizations
    : public ngraph::pass::GraphRewrite
{
public:
    CPUPostLayoutOptimizations()
        : GraphRewrite()
    {
        construct_weight_fusion();
        construct_slice_convertLayout_fusion();
        construct_reshape_convertLayout_fusion();
    }
    void construct_weight_fusion();
    void construct_slice_convertLayout_fusion();
    void construct_reshape_convertLayout_fusion();
};
