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
        /// This pass creates CompiledKernel ops enclosing sub-graphs that will be compiled and
        /// executed by MLIR.
        // TODO: WIP. Currently we only create a single CompiledKernel op for the whole function
        // body.
        class MLIRSubgraphExtractionPass : public ngraph::pass::FunctionPass
        {
        public:
            MLIRSubgraphExtractionPass() {}
            bool run_on_function(std::shared_ptr<Function> func) override;
            /// Checks if an ngraph node is supported by MLIR backend
            bool is_supported_mlir_op(std::shared_ptr<Node> node);

        private:
            static const std::set<std::type_index> m_supported_ops;
        };
    }
}
