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

#include "ng_op_fusion.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"

#include <llvm/ADT/SetVector.h>
#include <llvm/Support/Debug.h>

#define PASS_NAME "ng-op-fusion"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;
using namespace llvm;

namespace
{
    /// Kinds of operations in the context of fusion.
    enum OpFusionKind
    {
        // Includes element-wise operations, such as add, sub, mul, relu, etc.
        ElementWise = 0,
        // Includes more complex operations, such as dot and convolution.
        Complex,
        // Includes all the operations not fitting into any of the previous categories.
        Unknown
    };

    // TODO: Probably we could add a 'getOpFusionKind' interface that returns the kind of op in the
    // context of op fusion.
    static OpFusionKind getOpFusionKind(Operation* op)
    {
        if (isa<NGAddOp>(op) || isa<NGSubOp>(op) || isa<NGMulOp>(op) || isa<NGReluOp>(op))
            return OpFusionKind::ElementWise;
        if (isa<NGDotOp>(op))
            return OpFusionKind::Complex;
        return OpFusionKind::Unknown;
    }

    /// Container that represents a set of ops to be fused together.
    struct OpFusionGroup
    {
        OpFusionGroup(Operation* op)
            : kind(getOpFusionKind(op))
        {
            fusedOps.insert(op);
        }

        SmallSetVector<Operation*, 8> fusedOps;
        OpFusionKind kind;
    };

    /// Op-based fusion pass that makes fusion decision by taking advantage of nGraph ops' high
    /// level information. The algorithm classifies nGraph ops into multiple categories and applies
    /// op fusion following well-defined heuristics based on those categories. Currently, it only
    /// prints the fused ops into the dbgs output.
    ///
    /// Current TODOs:
    ///   * Support fusing ops with multiple uses.
    ///   * Consider fusion beyond the linear order of operations in the basic block.
    ///   * Support upcoming region ops.
    ///   * Do not treat constants as definitions that might prevent fusion.
    ///   * Model memory cost, etc.
    class NgOpFusionPass : public FunctionPass<NgOpFusionPass>
    {
    public:
        void runOnFunction() override;

    private:
        /// TODO
        OpFusionGroup* createNewGroup(Operation* op);

        /// TODO
        SmallVector<OpFusionGroup, 16> fusionGroups;
    };
} // namespace

static bool fuseOpIntoGroup(Operation* op, OpFusionGroup* currGroup)
{
    // Multiple uses not supported yet.
    if (!currGroup || !op->hasOneUse())
    {
        return false;
    }

    // Try to fuse op with the current group only if there is a producer-consumer relationship
    // between them.
    auto operands = op->getOperands();
    if (!std::any_of(operands.begin(), operands.end(), [&](Value val) {
            auto* operand = val.getDefiningOp();
            if (operand && currGroup->fusedOps.count(operand))
            {
                return true;
            }
            return false;
        }))
    {
        return false;
    }

    // Op fusion heuristics.
    auto opKind = getOpFusionKind(op);
    if (opKind == OpFusionKind::Unknown)
        return false;

    if ((currGroup->kind == OpFusionKind::ElementWise && opKind == OpFusionKind::ElementWise) ||
        (currGroup->kind == OpFusionKind::Complex && opKind == OpFusionKind::ElementWise))
    {
        currGroup->fusedOps.insert(op);
        return true;
    }

    return false;
}

OpFusionGroup* NgOpFusionPass::createNewGroup(Operation* op)
{
    // Multiple uses not supported yet.
    if (op->hasOneUse())
    {
        fusionGroups.push_back(OpFusionGroup(op));
        return &fusionGroups.back();
    }

    return nullptr;
}

void NgOpFusionPass::runOnFunction()
{
    OpFusionGroup* currGroup = nullptr;
    getFunction().walk([&](Operation* op) {
        if (fuseOpIntoGroup(op, currGroup))
            return;
        currGroup = createNewGroup(op);
    });

    unsigned i = 0;
    for (auto& group : fusionGroups)
    {
        if (group.fusedOps.size() > 1)
        {
            llvm::dbgs() << "Group " << i << "\n";
            llvm::dbgs() << "  Kind: " << group.kind << "\n";
            llvm::dbgs() << "  Ops:\n";
            for (auto* op : group.fusedOps)
            {
                op->dump();
            }
            llvm::dbgs() << "\n";
        }
        ++i;
    }
}

std::unique_ptr<mlir::Pass> mlir::createNgOpFusionPass()
{
    return std::make_unique<NgOpFusionPass>();
}

static PassRegistration<NgOpFusionPass> pass(PASS_NAME,
                                             "Enable op fusion optimization in nGraph dialect");
