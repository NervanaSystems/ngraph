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
#include "contrib/mlir/backend/pass/affine_lowerer.hpp"
#include "contrib/mlir/backend/pass/memory_optimization.hpp"
#include "contrib/mlir/utils.hpp"
#include "ngraph/check.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "mlir-cpu-backend"

// *** Optimization flags ***

static llvm::cl::opt<bool> clEnableNgInPlaceMemoryOpt(
    "ng-inplace-mem-opt",
    llvm::cl::init(false),
    llvm::cl::desc("Enable ngraph dialect in-place memory optimization pass"));

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

using namespace ngraph::runtime::ngmlir;

// Default optimization level.
llvm::CodeGenOpt::Level MLIRCPUBackend::mlirOptLevel = llvm::CodeGenOpt::Level::Aggressive;

std::unique_ptr<llvm::TargetMachine> MLIRCPUBackend::targetMachine;

bool MLIRCPUBackend::initialized = false;

/// Creates target machine for current host.
static llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
    createDefaultTargetMachine(unsigned optLevel)
{
    auto machineBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!machineBuilder)
    {
        return machineBuilder.takeError();
    }

    // Relocation model and code model are kept to default values. CodeGen optimization level
    // matches LLVM recommendations, i.e.:
    // enum Level {
    //   None,        // -O0
    //   Less,        // -O1
    //   Default,     // -O2, -Os
    //   Aggressive   // -O3
    // };
    machineBuilder->setCodeGenOptLevel((llvm::CodeGenOpt::Level)optLevel);
    return machineBuilder->createTargetMachine();
}

/// Returns the cache level size from `targetInfo` for the `cacheLevel` provided. If `userCacheSize`
/// is not zero, it returns `userCacheSize`.
static unsigned getCacheLevelSize(llvm::TargetTransformInfo& targetInfo,
                                  unsigned cacheLevel,
                                  unsigned userCacheSize)
{
    if (userCacheSize)
    {
        return userCacheSize;
    }

    llvm::Optional<unsigned> optCacheLevelSize;
    switch (cacheLevel)
    {
    case 1:
        optCacheLevelSize = targetInfo.getCacheSize(llvm::TargetTransformInfo::CacheLevel::L1D);
        break;
    case 2:
        optCacheLevelSize = targetInfo.getCacheSize(llvm::TargetTransformInfo::CacheLevel::L2D);
        break;
    default:
        NGRAPH_UNREACHABLE("Unsupported cache level: ", cacheLevel, ". Only 1 and 2 are supported");
    }

    NGRAPH_CHECK(optCacheLevelSize.hasValue() && "Cache level size is not available in TTI");
    return optCacheLevelSize.getValue();
}

void MLIRCPUBackend::init()
{
    // Mutex to safely initialize CPU backend
    static std::mutex mlirInitMutex;

    std::unique_lock<std::mutex> lock(mlirInitMutex);

    if (!initialized)
    {
        // Override default optimization level with macro value.
        if (char* optLevelStr = std::getenv("NGRAPH_MLIR_OPT_LEVEL"))
        {
            unsigned clOptLevel = std::stoi(optLevelStr);
            NGRAPH_CHECK(clOptLevel >= 0 && clOptLevel <= 3, "Invalid optimization level");
            mlirOptLevel = (llvm::CodeGenOpt::Level)clOptLevel;
        }

        // Initialize LLVM targets and target machine for current host.
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        auto expectedTargetMachine = createDefaultTargetMachine(mlirOptLevel);
        NGRAPH_CHECK(expectedTargetMachine, "Invalid target machine");
        targetMachine = std::move(*expectedTargetMachine);

        initialized = true;
    }
}

void MLIRCPUBackend::codegen()
{
    optimizeNgDialect();
    lowerNgDialect();
}

void MLIRCPUBackend::lowerNgDialect()
{
    // Lower NG dialect to Affine
    mlir::PassManager pm(&m_context);
    pm.addPass(mlir::createDialectLoweringPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Incorrect module after dialect lowering");
    }

    optimizeAffineDialect();

    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    lowerStandardDialect();
}

// Lower Standard dialect to LLVM dialect
void MLIRCPUBackend::lowerStandardDialect()
{
    mlir::PassManager pm(&m_context);
    pm.addPass(mlir::createLowerToLLVMPass());

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Incorrect module after dialect lowering");
    }
}

// Receives affine dialect as input and applies affine and standard dialect based optimizations.
// Lowering from affine dialect to standard dialect happens along the way. Output consists of
// standard dialect only ops.
void MLIRCPUBackend::optimizeAffineDialect()
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

void MLIRCPUBackend::optimizeNgDialect()
{
    mlir::PassManager pm(&m_context);
    mlir::applyPassManagerCLOptions(pm);
    if (clEnableNgInPlaceMemoryOpt)
    {
        pm.addPass(mlir::createMemoryOptimizationPass());
    }

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }
}
