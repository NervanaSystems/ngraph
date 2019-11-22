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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#include "utils.hpp"

#include "contrib/mlir/core/ngraph_dialect/dialect.hpp"

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>

static llvm::cl::opt<bool> clPrintIRAfterAll(
    "ngraph-print-ir-after-all",
    llvm::cl::init(false),
    llvm::cl::desc(
        "Print IR after transformation that are not implemented as passes in the MLIRCompiler. It "
        "complements MLIR -print-ir-after-all and LLVM -print-after-all flags"));

void ngraph::runtime::ngmlir::initializeNGraphMLIR()
{
    // Initialize a dialect only once.
    // We currently have no way to query if a dialect is previously
    // registered. So using a global flag instead.
    static bool init = false;
    if (!init)
    {
        mlir::registerDialect<mlir::NGraphOpsDialect>();
        init = true;
    }
}

void ngraph::runtime::ngmlir::dumpMlirModule(const std::string msg, mlir::ModuleOp module)
{
    if (clPrintIRAfterAll)
    {
        llvm::dbgs() << "*** IR Dump After " << msg << " ***\n";
        module.dump();
        llvm::dbgs() << "\n\n";
    }
}
