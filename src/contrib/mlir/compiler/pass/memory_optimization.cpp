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

#include "contrib/mlir/compiler/compiler.hpp"
#include "contrib/mlir/compiler/dialect/ops.hpp"
#include "contrib/mlir/compiler/dialect/type.hpp"

#include "ngraph/assertion.hpp"

#include <llvm/ADT/DenseSet.h>
#include <map>
#include <mlir/EDSC/Builders.h>
#include <mlir/EDSC/Helpers.h>
#include <mlir/EDSC/Intrinsics.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

// anonymous namespace
// no need to expose any of the following outside of this file
namespace
{
    using namespace ngraph::runtime;
    using namespace ngraph::runtime::ngmlir;
    using namespace mlir;

    // A helper data-structure to track cannot alias relationship between tensor syms
    // If NoAlias[T] contains S, then T and S cannot alias. 
    // The relationship is transitive and we compute transitive closure on each update to the 
    // the strucutre. 
    class AliasMap {

        public:
        AliasMap();
        /// Checks if values a and b can alias
        bool canAlias(Value* a, Value* b);
        bool insertNoAlias(Value* a, Value* b);

        private: 
        void computeClosure();
        private:
        using Row = SmallVector<int8_t,4>;
        std::unordered_map<Value*, unsigned> m_valueToIdx;
        SmallVector<Row, 4> m_reachability;
        unsigned m_maxIdx;
        bool m_needsTransitiveClosure;
    };

    /// Memory Optimization pass
    /// - Tries to perform operations in place where applicable by assigning a virtual buffer ID
    ///    to values. Those are used later in affine lowering pass to create or re-use memrefs
    class MemoryOptimizationPass : public mlir::FunctionPass<MemoryOptimizationPass>
    {
    public:
        MemoryOptimizationPass()
        {
            m_inplaceOps = {
#define MLIR_OP(OP, INPLACE) {OP::getOperationName().str(), INPLACE},
#include "contrib/mlir/compiler/op_lowerers.inc"
            };
        }
        void runOnFunction() override;

    private:
        bool isSafeInPlace(mlir::Operation* op);
        std::unordered_map<std::string, bool> m_inplaceOps;
        static unsigned bufferId;
    };

    unsigned MemoryOptimizationPass::bufferId = 0;

    void MemoryOptimizationPass::runOnFunction()
    {
        auto f = getFunction();

        f.walk([&](mlir::Operation* op) {
            if (!isSafeInPlace(op))
            {
                return;
            }

            if (op->getNumResults() > 1)
            {
                return;
            }

            auto defVal = op->getResult(0);

            // If the defined value is an output of the sub-graph, cannot do it in place
            for (auto use = defVal->use_begin(); use != defVal->use_end(); use++)
            {
                auto useOp = use->getOwner();
                if (isa<NGReturnOp>(useOp))
                {
                    return;
                }
            }

            // Check if we can re-use the buffer of any of the inputs. Conjunction of the following:
            // - single use value or all uses in the current op
            // - not an input argument

            // TODO: Check instead if last post-dominating (dataflow-wise) use.
            for (auto opnd = op->operand_begin(); opnd != op->operand_end(); opnd++)
            {
                auto val = *opnd;
                // we optimize if the val has one use or if all uses are in the current op
                bool optimize;

                optimize = val->hasOneUse();

                if (!optimize)
                {
                    optimize = true;
                    // check if all uses are in the current op
                    for (auto use = val->use_begin(); use != val->use_end(); use++)
                    {
                        if (use->getOwner() != op)
                        {
                            optimize = false;
                        }
                    }
                }

                if (optimize)
                {
                    // do we have a buffer id attached to this value
                    auto defOp = val->getDefiningOp();
                    // If no defining op, then this is a block arg, skip operand
                    if (!defOp)
                    {
                        continue;
                    }
                    IntegerAttr attr = getBufferId(defOp);

                    if (!attr)
                    {
                        // attach a new buffer id
                        attr = setBufferId(defOp, this->bufferId++);
                    }
                    // propagate attribute to dst, and we are done
                    setBufferId(op, attr);

                    return;
                }
            }
        });
    }

    bool MemoryOptimizationPass::isSafeInPlace(mlir::Operation* op)
    {
        auto it = m_inplaceOps.find(op->getName().getStringRef().str());
        return it != m_inplaceOps.end() ? it->second : false;
    }


    AliasMap::AliasMap()
    {
        for (auto i = 0; i < m_reachability.size(); i++)
        {
            auto& v = m_reachability[i];
            std::fill(v.begin(), v.end(), 0);
        }
        m_maxIdx = 0;
        m_needsTransitiveClosure = false;
    }
    bool AliasMap::canAlias(Value* a, Value* b)
    {
        NGRAPH_CHECK(m_needsTransitiveClosure == false, "Relationship needs transitive closure");
        auto a_it = m_valueToIdx.find(a);
        auto b_it = m_valueToIdx.find(b);
        if (a_it == m_valueToIdx.end() || b_it == m_valueToIdx.end())
        {
            // at least one value doesn't exist in the cannot-alias relationship
            return true;
        }
        auto a_idx = *a_it;
        auto b_idx = *b_it;
        return !m_reachability[a_idx][b_idx] && m_reachability[b_idx][a_idx]);
    }
    void AliasMap::insertNoAlias(Value* a, Value* b)
    {
        m_needsTransitiveClosure = true;
        int rowsToAdd = 0;
        auto it = m_valueToIdx.find(a);
        if (it == m_valueToIdx.end())
        {
            rowsToAdd++;
            m_valueToIdx[a] = m_maxIdx++;
        }

        it = m_valueToIdx.find(b);
        if (it == m_valueToIdx.end())
        {
            rowsToAdd++;
            m_valueToIdx[b] = m_maxIdx++;
        }

        if (rowsToAdd)
        {
            // expand existing rows with 1 or 2 additional columns
            for (auto i = 0; i < m_maxIdx - rowsToAdd; i++)
            {
                m_reachability[i].push_back(0);
                if (rowsToAdd > 1)
                {
                    m_reachability[i].push_back(0);
                }
            }
            // add 1 or 2 additional rows
            m_reachability.push_back(Row(m_maxIdx, 0));
            if (rowsToAdd > 1)
            {
                m_reachability.push_back(Row(m_maxIdx, 0));
            }
        }

#ifdef NGRAPH_DEBUG_ENABLE
        checkInvariance();
#endif
        computeTransitiveClosure();
    }

    void AliasMap::computeTransitiveClosure()
    {
        for (unsigned k = 0; k < m_maxIdx; k++)
        {
            for (unsigned i = 0; i < m_maxIdx; i++)
            {
                for (unsigned j = 0; j < m_maxIdx; j++)
                {
                    if (m_reachability[i][k] && m_reachability[k][j])
                    {
                        m_reachability[i][j] = 1;
                    }
                }
            }
        }
            
    }

    void AliasMap::checkInvariance() 
    {
        NGRAPH_CHECK(m_reachability.size() == m_maxIdx);
        for (auto& v : m_reachability)
        {
            NGRAPH_CHECK(v.size() == m_maxIdx, "Non-square matrix");
        }

        for (unsigned i = 0; i < m_maxIdx; i++)
        {
            for (unsigned j = 0; j < m_maxIdx; j++)
            {
                NGRAPH_CHECK(m_reachability[i][j] == m_reachability[j][i], "Non-symmetric relationship");
            }
        }
    }
}




namespace mlir
{
    std::unique_ptr<Pass> createMemoryOptimizationPass()
    {
        return std::make_unique<MemoryOptimizationPass>();
    }
} // namespace mlir
