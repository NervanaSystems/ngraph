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

#include "ngraph/function.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace intelgpu
        {
            // This function writes the input func into file in Graphviz format.
            // On large graphs, the "dot" utility requires lot of time to visualize the input.
            // Example: dot -Tpdf intelgpu_Function_0_orig.dot -o intelgpu_Function_0_orig.pdf
            void visualize_tree(const std::shared_ptr<Function>& func,
                                const std::string& file_prefix,
                                const std::string& file_suffix);
        }
    }
}
