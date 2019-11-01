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


#pragma once

#include <mutex>
#include "ngraph/pass/pass.hpp"

#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "ngraph/type/element_type.hpp"

#include "contrib/mlir/utils.hpp"

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

#include <mlir/Analysis/Verifier.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/Pass.h>

#include <memory>

namespace ngraph
{
    namespace pass
    {
        class NgDialectConversionPass : public mlir::ModulePass<ngraph::pass::NgDialectConversionPass> {
        public:
            NgDialectConversionPass(const ngraph::op::CompiledKernel* compiledKernel);
	    void buildNgDialect(); 
	    void runOnModule() override;

        private:
            const ngraph::op::CompiledKernel* m_compiledKernel;
        };
	std::unique_ptr<mlir::Pass> CreateNgDialectConversionPass(const ngraph::op::CompiledKernel* compiledKernel);
    }
}
