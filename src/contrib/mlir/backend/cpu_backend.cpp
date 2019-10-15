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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "cpu_backend.hpp"
#include "ngraph/check.hpp"
#include "contrib/mlir/backend/pass/affine_lowerer.hpp"

// TODO: Clean up unneeded files
#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>


static llvm::cl::opt<bool>
    clEnableAffineLoopFusion("ngraph-affine-loop-fusion",
                             llvm::cl::init(false),
                             llvm::cl::desc("Enable loop fusion optimization in Affine dialect"));

static llvm::cl::opt<bool>
    clEnableAffineLoopTiling("ngraph-affine-loop-tile",
                             llvm::cl::init(false),
                             llvm::cl::desc("Enable loop tiling optimization in Affine dialect"));

static llvm::cl::opt<unsigned>
    clLoopTilingCacheLevel("ngraph-affine-loop-tile-cache-level",
                           llvm::cl::init(2),
                           llvm::cl::desc("Cache level to which to apply affine loop tiling."));

static llvm::cl::opt<unsigned> clLoopTilingCacheSize(
    "ngraph-affine-loop-tile-cache-size",
    llvm::cl::init(0),
    llvm::cl::desc(
        "Cache size to use in affine loop tiling. If not zero, it overrides the cache-size "
        "inferred from the host CPU using for the cache level specified by "
        "-ngraph-loop-tile-cache-level."));

// *** Debug flags ***

static llvm::cl::opt<bool>
    clDumpObjectFile("ngraph-dump-mlir-object-file",
                     llvm::cl::desc("Dump MLIR JITted-compiled object to file specified with "
                                    "-object-filename (<input file>.o by default)."));

static llvm::cl::opt<std::string>
    clObjectFilename("ngraph-mlir-object-filename",
                     llvm::cl::desc("Dump MLIR JITted-compiled object to file jitted_mlir.o"));

using namespace ngraph::runtime::ngmlir;



void MLIRCPUBackend::codegen()
{
    lowerNgDialect();

}

// Lowers nGraph dialect all the way to LLVM module.
void MLIRCPUBackend::lowerNgDialect()
{
    // Lower NG dialect to Affine
    mlir::PassManager pm(&m_context);
    pm.addPass(mlir::createDialectLoweringPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    if (mlir::failed(pm.run(m_module.get()))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Incorrect module after dialect lowering");
    }

    optimize();

    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    // Lower Standard dialect to LLVM dialect.
    mlir::LLVMTypeConverter llvmConverter(&m_context);
    mlir::OwningRewritePatternList patterns;
    mlir::populateLoopToStdConversionPatterns(patterns, &m_context);
    mlir::populateStdToLLVMConversionPatterns(llvmConverter, patterns);

    mlir::ConversionTarget target(m_context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>(
        [&](mlir::FuncOp op) { return llvmConverter.isSignatureLegal(op.getType()); });
    auto result = mlir::applyFullConversion(m_module, target, std::move(patterns), &llvmConverter);
    NGRAPH_CHECK(succeeded(result), "Standard to LLVM dialect conversion failed");
    // TODO: Enable after moving to utils
    //dumpMlirModule("LLVM-IR Dialect Conversion");
}

// Receives affine dialect as input and applies affine and standard dialect based optimizations.
// Lowering from affine dialect to standard dialect happens along the way. Output consists of
// standard dialect only ops.
void MLIRCPUBackend::optimize()
{
    // Create target transform info to obtain some target information to be used in MLIR
    // optimizations. This is a temporary attempt to retrieve some target information by reusing
    // LLVM TTI infra while MLIR does not have target model.
    llvm::LLVMContext llvmContext;
    auto module = std::unique_ptr<llvm::Module>(new llvm::Module("test", llvmContext));
    module->setDataLayout(targetMachine->createDataLayout());
    auto ttiSetupFunc = llvm::cast<llvm::Function>(
        module
            ->getOrInsertFunction("__ngraph_tti_setup",
                                  llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), {}))
            .getCallee());
    auto targetInfo = targetMachine->getTargetTransformInfo(*ttiSetupFunc);

    // Populate pass manager with affine dialect optimizations.
    mlir::PassManager pm(&m_context);
    if (clEnableAffineLoopFusion)
    {
        pm.addPass(mlir::createLoopFusionPass());
    }

    if (clEnableAffineLoopTiling)
    {
        unsigned cacheLevelSize =
            getCacheLevelSize(targetInfo, clLoopTilingCacheLevel, clLoopTilingCacheSize);
        LLVM_DEBUG(llvm::dbgs() << "Enabling Affine Loop Tiling for cache level "
                                << clLoopTilingCacheLevel
                                << ": "
                                << cacheLevelSize
                                << " bytes.\n");
        pm.addPass(mlir::createLoopTilingPass(cacheLevelSize));
    }

    // Populate pass manager with affine dialect to Std dialect conversion.
    pm.addPass(mlir::createLowerAffinePass());

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    // Run pass manager passes.
    auto result = pm.run(m_module.get());
    NGRAPH_CHECK(succeeded(result), "Affine optimizaitons and convertion to Std dialect failed");

    // Run Std dialect optimizations.
    // TODO
}
