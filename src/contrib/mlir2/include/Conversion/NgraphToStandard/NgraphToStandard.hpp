//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#ifndef MLIR_NGRAPH_PASSES_H
#define MLIR_NGRAPH_PASSES_H

#include <memory>

namespace mlir
{
    class Pass;

    namespace ngraph
    {
        /// Create a pass for lowering operations the remaining `Ngraph` operations, as
        /// well as `Affine` and `Std`, to the LLVM dialect for codegen.
        std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
    }
}

#endif // MLIR_NGRAPH_PASSES_H
