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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming
// convention.

#include "compiler.hpp"
#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <memory>
#include <mlir/Conversion/SCFToStandard/SCFToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>
#include <mutex>
#include "contrib/mlir/utils.hpp"
#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/node.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph_dialect/dialect.hpp"
#include "ngraph_dialect/ops.hpp"
#include "ngraph_dialect/type.hpp"
#include "pass/ng_dialect_builder.hpp"
#include "pass/ng_dialect_fused_ops.hpp"
#include "pass/ng_op_fusion.hpp"

// Defines a new LLVM debug type for this file to be used by LLVM_DEBUG macro.
#define DEBUG_TYPE "mlir-compiler"

static llvm::cl::opt<bool> clEnableNgKernelLibFusion(
    "ngraph-kernel-lib-fusion",
    llvm::cl::init(false),
    llvm::cl::desc("Enable the ngraph pass that fuses ops to use kernel library"));

using llvm::ArrayRef;
using llvm::BumpPtrAllocator;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::StringSaver;

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

static llvm::cl::opt<bool> clEnableOpFusion("ngraph-op-fusion",
                                            llvm::cl::init(false),
                                            llvm::cl::desc("Enable ngraph dialect op fusion pass"));

bool MLIRCompiler::initialized = false;

MLIRCompiler::MLIRCompiler(std::shared_ptr<ngraph::Function> function, ::mlir::MLIRContext& context)
    : m_function(function)
    , m_context(context)
{
    NGRAPH_CHECK(initialized, "Cannot instantiate a compiler without initializing MLIR");
}

void MLIRCompiler::init()
{
    // Mutex to safely initialize MLIR.
    static std::mutex mlirInitMutex;

    std::unique_lock<std::mutex> lock(mlirInitMutex);

    if (!initialized)
    {
        // TODO: Remove this as it is not part of compiler init
        initializeNGraphMLIR();

        // Register MLIR command line options in the pool of supported flags and and
        // process flags
        // from environment variable to be used by nGraph, MLIR and LLVM.
        mlir::registerPassManagerCLOptions();
        // Emulate: llvm::cl::ParseEnvironmentOptions("ngraph", "NGRAPH_MLIR_OPTIONS", "");
        llvm::Optional<std::string> envValue =
            llvm::sys::Process::GetEnv(StringRef("NGRAPH_MLIR_OPTIONS"));
        if (envValue)
        {
            SmallVector<const char*, 20> newArgv;
            BumpPtrAllocator A;
            StringSaver Saver(A);
            newArgv.push_back(Saver.save("ngraph").data());
            llvm::cl::TokenizeGNUCommandLine(*envValue, Saver, newArgv);
            int newArgc = static_cast<int>(newArgv.size());
            llvm::cl::ParseCommandLineOptions(newArgc, &newArgv[0], StringRef(""));
        }
        initialized = true;
    }
}

void MLIRCompiler::compile()
{
    buildNgDialectModule();
}

// Creates an MLIR module and function with nGraph dialect ops from the input
// CompiledKernel.
void MLIRCompiler::buildNgDialectModule()
{
    // initialize an empty module
    m_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&m_context));

    mlir::PassManager pm(&m_context);
    pm.addPass(ngraph::pass::createNgDialectConversionPass(m_function, &m_context));

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Invalid module after lowering to NG dialect");
    }

    dumpMlirModule("nGraph Dialect Construction", m_module.get());

    optimizeNgDialect();
}

void MLIRCompiler::optimizeNgDialect()
{
    mlir::PassManager pm(&m_context);
    if (clEnableNgKernelLibFusion)
    {
        pm.addPass(ngraph::pass::createNgDialectFusedOpsPass());
    }

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Invalid module after NG dialect optimization");
    }

    dumpMlirModule("nGraph Dialect optimization", m_module.get());
}
