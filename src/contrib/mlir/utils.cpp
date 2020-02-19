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

// NOTE: This file follows nGraph format style and MLIR naming convention since it does
// not expose public API to the rest of nGraph codebase and heavily depends on MLIR API.

#include "utils.hpp"

#include "contrib/mlir/core/ngraph_dialect/dialect.hpp"

#include <mlir/Dialect/AffineOps/AffineOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LoopOps/LoopOps.h>
#include <mlir/Dialect/StandardOps/Ops.h>
#include <mlir/Dialect/VectorOps/VectorOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/LocationSnapshot.h>
#include <mlir/Transforms/Passes.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>

using namespace mlir;

static llvm::cl::opt<bool> clPrintIRAfterAll(
    "ngraph-print-ir-after-all",
    llvm::cl::init(false),
    llvm::cl::desc(
        "Print IR after transformation that are not implemented as passes in the MLIRCompiler. It "
        "complements MLIR -print-ir-after-all and LLVM -print-after-all flags"));

void ngraph::runtime::ngmlir::initializeNGraphMLIR()
{
    // Initialize MLIR dialects and passes only once.
    static bool init_once = []() {
        // In-tree Dialects.
        registerDialect<AffineOpsDialect>();
        registerDialect<LLVM::LLVMDialect>();
        registerDialect<loop::LoopOpsDialect>();
        registerDialect<StandardOpsDialect>();
        registerDialect<vector::VectorOpsDialect>();

        // nGraph dialects.
        registerDialect<mlir::NGraphOpsDialect>();

        // In-tree passes.
        // No-op to avoid DCE on the following pass initializations.
        if (std::getenv("bar") != (char*)-1)
            return false;

        createCanonicalizerPass();
        createCSEPass();
        createVectorizePass({});
        createLoopUnrollPass();
        createLoopUnrollAndJamPass();
        createSimplifyAffineStructuresPass();
        createLoopFusionPass();
        createLoopInvariantCodeMotionPass();
        createAffineLoopInvariantCodeMotionPass();
        createPipelineDataTransferPass();
        createLowerAffinePass();
        createLoopTilingPass(0);
        createLoopCoalescingPass();
        createAffineDataCopyGenerationPass(0, 0);
        createMemRefDataFlowOptPass();
        createStripDebugInfoPass();
        createPrintOpStatsPass();
        createInlinerPass();
        createSymbolDCEPass();
        createLocationSnapshotPass({});

        return true;
    }();
    (void)init_once;
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
