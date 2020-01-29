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

#include "ng_dialect_fused_ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/dialect.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/type.hpp"

#include <llvm/IR/Module.h>
#include <mlir/EDSC/Builders.h>
#include <mlir/EDSC/Helpers.h>
#include <mlir/EDSC/Intrinsics.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <iostream>
#define DEBUG_TYPE "mlir-compiler"
#define PASS_NAME "ngraph_dialect_fusion"

using llvm::SmallVector;
using llvm::StringRef;
using llvm::ArrayRef;

using namespace ngraph;
using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::op;

namespace mlir
{
    static Value createSgemmOp(
        PatternRewriter& rewriter, Operation* old_op, Value input1, Value input2, Value input3)
    {
        auto castedOp0 = dyn_cast_or_null<NGAddOp>(old_op);
        SmallVector<Value, 4> values;
        SmallVector<NamedAttribute, 4> attrs;
        values.push_back(input1);
        values.push_back(input2);
        values.push_back(input3);
        attrs.emplace_back(
            rewriter.getIdentifier("alpha"),
            rewriter.getFloatAttr(mlir::Builder(rewriter.getContext()).getF32Type(), 1.0));
        attrs.emplace_back(
            rewriter.getIdentifier("beta"),
            rewriter.getFloatAttr(mlir::Builder(rewriter.getContext()).getF32Type(), 1.0));
        attrs.emplace_back(rewriter.getIdentifier("transA"), rewriter.getBoolAttr(false));
        attrs.emplace_back(rewriter.getIdentifier("transB"), rewriter.getBoolAttr(false));
        SmallVector<Type, 4> types;
        for (auto v : castedOp0.getODSResults(0))
        {
            types.push_back(v.getType());
        }
        return rewriter.create<NGGemmOp>(castedOp0.getLoc(), types, values, attrs);
    }

#include "fused_ops_pattern.h.inc"
}
namespace
{
    class NgDialectFusedOpsPass : public mlir::ModulePass<NgDialectFusedOpsPass>
    {
    public:
        NgDialectFusedOpsPass(mlir::ModuleOp& module, mlir::MLIRContext* context)
            : m_module(module)
            , m_context(context)
        {
        }

        NgDialectFusedOpsPass(const NgDialectFusedOpsPass& obj);

    private:
        void runOnModule() override;
        mlir::ModuleOp m_module;
        mlir::MLIRContext* m_context;
    };
}

NgDialectFusedOpsPass::NgDialectFusedOpsPass(const NgDialectFusedOpsPass& obj)
    : m_module(obj.m_module)
    , m_context(obj.m_context)
{
}

void NgDialectFusedOpsPass::runOnModule()
{
    OwningRewritePatternList patterns;
    mlir::populateWithGenerated(m_context, &patterns);

    // Gather functions to be processed. Note that new functions will be added to module as part
    // of the function signature conversion so we have to collect the original ones before hand.
    SmallVector<FuncOp, 2> origFuncOps(m_module.getOps<FuncOp>());

    for (auto origFunc : origFuncOps)
    {
        applyPatternsGreedily(origFunc, patterns);
    }
}

std::unique_ptr<Pass> ngraph::pass::createNgDialectFusedOpsPass(mlir::ModuleOp module,
                                                                mlir::MLIRContext* context)
{
    return std::make_unique<NgDialectFusedOpsPass>(module, context);
}
