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

#include <llvm/Support/Debug.h>

#define DEBUG_TYPE "ng-op-fusion"

using namespace mlir;

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
            fusedOps.push_back(op);
        }

        SmallVector<Operation*, 8> fusedOps;
        OpFusionKind kind;
    };

    /// Op-based fusion pass that makes fusion decision by taking advantage of nGraph ops' high
    /// level information. The algorithm classifies nGraph ops into multiple categories and applies
    /// op fusion following well-defined heuristics based on those categories. Currently, it only
    /// prints the fused ops into the dbgs output.
    ///
    /// Current TODOs:
    ///   * Take def-use infor into account.
    ///   * Take number of uses of a particular op into account.
    ///   * Take shapes into account.
    ///   * Consider fusion beyond the linear order of operations in the basic bloc.
    ///   * Support upcoming region ops.
    ///   * Do not treat constants as definitions that might prevent fusion.
    ///   * Model memory cost, etc.
    class NgOpFusionPass : public FunctionPass<NgOpFusionPass>
    {
    public:
        void runOnFunction() override;

    private:
        /// TODO
        bool fuseIntoActiveGroup(Operation* op);
        void createNewGroup(Operation* op);

        /// TODO
        SmallVector<OpFusionGroup, 16> fusionGroups;
    };
} // namespace

bool NgOpFusionPass::fuseIntoActiveGroup(Operation* op)
{
    if (fusionGroups.empty())
    {
        // No active group.
        return false;
    }

    // Op fusion heuristics.
    auto& group = fusionGroups.back();
    auto opKind = getOpFusionKind(op);
    if (opKind == OpFusionKind::Unknown)
        return false;

    if ((group.kind == OpFusionKind::ElementWise && opKind == OpFusionKind::ElementWise) ||
        (group.kind == OpFusionKind::Complex && opKind == OpFusionKind::ElementWise))
    {
        group.fusedOps.push_back(op);
        return true;
    }

    return false;
}

void NgOpFusionPass::createNewGroup(Operation* op)
{
    fusionGroups.push_back(OpFusionGroup(op));
}

void NgOpFusionPass::runOnFunction()
{
    getFunction().walk([&](Operation* op) {
        if (fuseIntoActiveGroup(op))
            return;
        createNewGroup(op);
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
