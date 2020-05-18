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
#include <mlir/Dialect/AffineOps/EDSC/Builders.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <iostream>

using llvm::SmallVector;
using llvm::StringRef;
using llvm::ArrayRef;

using namespace ngraph;
using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::op;

#define PASS_NAME "fuse-ngraph-dialect"
#define DEBUG_TYPE PASS_NAME

namespace mlir
{
    static Value createSgemmOp(
        PatternRewriter& rewriter, Operation* old_op, Value input1, Value input2, Value input3)
    {
        auto castedOp0 = dyn_cast_or_null<NGAddOp>(old_op);
        SmallVector<Value, 4> values{input1, input2, input3};
        SmallVector<NamedAttribute, 4> attrs;
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
        NgDialectFusedOpsPass() {}
    private:
        void runOnModule() override;
    };
}

void NgDialectFusedOpsPass::runOnModule()
{
    OwningRewritePatternList patterns;
    mlir::populateWithGenerated(&getContext(), &patterns);

    // Gather functions to be processed. Note that new functions will be added to module as part
    // of the function signature conversion so we have to collect the original ones before hand.
    SmallVector<FuncOp, 2> origFuncOps(getModule().getOps<FuncOp>());

    for (auto origFunc : origFuncOps)
    {
        applyPatternsGreedily(origFunc, patterns);
    }
}

std::unique_ptr<Pass> ngraph::pass::createNgDialectFusedOpsPass()
{
    return std::make_unique<NgDialectFusedOpsPass>();
}

static PassRegistration<NgDialectFusedOpsPass>
    pass(PASS_NAME, "Fuse ngraph dialct based on the pattern match");
