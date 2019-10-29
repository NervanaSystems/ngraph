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
    class AliasRelation 
    {

        public:
        AliasRelation();
        /// Checks if values a and b can alias
        bool canAlias(Value* a, Value* b);
        void insertNoAlias(Value* a, Value* b);

        private: 
        void computeTransitiveClosure();
#ifdef NGRAPH_DEBUG_ENABLE
        void checkInvariance();
#endif
        private:
        using Row = SmallVector<int8_t,10>;
        std::unordered_map<Value*, unsigned> m_valueToIdx;
        SmallVector<Row, 10> m_reachability;
        unsigned m_maxIdx; 
        bool m_needsTransitiveClosure;
    };

    class Liveness
    {
        public:
        bool isLive(Value* v);
        void setLive(Value* v);
        void kill(Value* v);
        void getLiveValues(llvm::SmallVectorImpl<Value*>& values);
        void reset();
        private: 
        unsigned m_maxIdx = 0;
        SmallVector<bool, 10> m_liveness;
        std::unordered_map<Value*, unsigned> m_valueToIdx;
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
        bool isInputOrOutputValue(mlir::Value* value);
        std::unordered_map<std::string, bool> m_inplaceOps;
        static unsigned bufferId;
    };

    unsigned MemoryOptimizationPass::bufferId = 0;

// Go backwards over instructions
//  Update aliasing assignment
//  Update liveness info
// 
// Aliasing conditions:
// General conditions:
//      Operand cannot be argument or output of the sub-graph
// Destructive in-place:
//      Find first operand where:
//          It is last use (operand is dead). If both are las-use, then pick one with lower # of uses. 
//      Assert operand has no prior buffer Id assigned (since last use). 
//      If result has bufferId + Offset, then copy them to operand. If not create new BufferId and Offset = 0. 
//
// Non-Destructive in-place:
//      Concat:
//          Concat axis is most-significant non-one axis. 
//          All operands can alias dest. 
//          Compute buffer ID and offset for each operand and try to assign, if any of the operands already have an attribute that doesnt match what we want to assign, bail out. 
//
//
//      Slice: TBD






    void MemoryOptimizationPass::runOnFunction()
    {
        Liveness liveness;
        AliasRelation aliasRelation;
        auto f = getFunction();

        auto& blocks = f.getBlocks();
        if (blocks.size() != 1)
        {
            // single block func for now
            return;
        }
        auto& block = *(blocks.begin());

        // scan instructions backwards
        for (auto it = block.rbegin(); it != block.rend(); it++)
        {
            Operation* op = &(*it);

            // TODO: replace with Op Interface check
            if (/*inplace and destructive*/ false)
            {
                NGRAPH_CHECK(op->getNumResults() == 1, "Destructive in-place with multi-def ?");
                Value* use = nullptr;

                unsigned useCount = 0;
                // pick a dead operand, not input or output with the least number of uses
                for (auto opnd : op->getOperands())
                {
                   if (!liveness.isLive(opnd) && !isInputOrOutputValue(opnd))
                   {
                       unsigned uses = 0;
                       for (auto& i : opnd->getUses())
                       {
                           uses++;
                       }
                       if (useCount > uses)
                       {
                           use = opnd;
                       }
                   }
                }
                if (use)
                {
                    // assign new buffer or copy buffer info from dst
                    IntegerAttr attr = getBufferId(op);
                    if (!attr)
                    {
                        // attach a new buffer id, and 0 offset on obth src and result
                        attr = setBufferId(op, this->bufferId++);
                        setBufferId(use->getDefiningOp(), attr);

                        attr = setBufferOffset(op, 0);
                        setBufferOffset(use->getDefiningOp(), 0);
                    }
                    else
                    {
                        // copy result buffer id and offset to src
                        setBufferId(use->getDefiningOp(), attr);
                        attr = getBufferOffset(op);
                        setBufferOffset(use->getDefiningOp(), attr);
                    }

                    // update aliasing info
                    // use value cannot alias any live value 
                    SmallVector<Value*, 10> liveValues;
                    liveness.getLiveValues(liveValues);
                    for (auto& value :  liveValues)
                    {
                        aliasRelation.insertNoAlias(use, value);
                    }
                    
                }
                
            } 

            // TODO: replace with Op Interface check
            if (/*inplace and non-destructive*/ false)
            {
                if (auto concat = cast<mlir::NGConcatOp>(op))
                {
                    // concat on the highest non-one axis
                    auto concatAxis = concat.concatenation_axis();
                    auto result = concat.getResult();
                    auto shape = (result.getType().cast<NGTensorType>()).getShape();
                    bool optimize = true;
                    std::vector<int> offsetDeltas;
                    IntegerAttr attr;
                    int bufferId = -1, baseOffset = 0;

                    for (auto i : shape)
                    {
                        if (i == concatAxis)
                        {
                            break;
                        }
                        if (i != 1)
                        {
                            optimize = false;
                        }
                    }
                    if (optimize)
                    {
                        // check that all operands and dst can alias
                        // and that none is input or output
                        for (auto opnd : op->getOperands())
                        {
                            if (!aliasRelation.canAlias(result, opnd) ||
                                isInputOrOutputValue(opnd))
                            {
                                optimize = false;
                            }
                        }
                    }

                    if (optimize)
                    {
                        // calculate offset deltas
                        int offsetDelta = 0;
                        for (auto i = 0; i < op->getNumOperands(); i++)
                        {
                            if (i = 0)
                            {
                                offsetDeltas.push_back(0);
                            }
                            else
                            {
                                auto opnd = op->getOperand(i-1);
                                auto tensorType = opnd->getType().cast<NGTensorType>();
                                offsetDelta += tensorType.getNumElements();
                                offsetDeltas.push_back(offsetDelta);
                            }
                        }
                        // check for consistent pre-existing buffer assignments
                        
                        // dest has an assignment
                        attr = getBufferId(op);
                        if (attr)
                        {
                            // set buffer ID and base offset to that of dest's
                            bufferId = attr.getInt();
                            baseOffset = getBufferOffset(op).getInt();

                            // check if we can re-use it for all src operands
                            int currOffset = 0;
                            for (auto i = 0; i < op->getNumOperands(); i++)
                            {
                                auto opnd = op->getOperand(i);
                                auto defOp = opnd->getDefiningOp();
                                // expected offset 
                                currOffset = baseOffset + offsetDeltas[i];

                                attr = getBufferId(defOp);
                                if (attr)
                                {
                                    if (attr.getInt() != bufferId || 
                                        currOffset != getBufferOffset(defOp).getInt())
                                    {
                                        // buffer ID mismatch, bailout
                                        optimize = false;
                                        break;
                                    }
                                }
                            }
                        }
                        else
                        {
                            // dst has no buffer assignment
                            // Check if any of the srcs does. If so, check if we can use it for all operands and dst
                            int currOffset;
                            bool srcFound = false;
                            for (auto i = 0; i < op->getNumOperands(); i++)
                            {
                                auto opnd = op->getOperand(i);
                                auto defOp = opnd->getDefiningOp();

                                attr = getBufferId(defOp);
                                if (attr)
                                {
                                    bufferId = attr.getInt();
                                    attr = getBufferOffset(defOp);
                                    baseOffset = attr.getInt() - offsetDeltas[i];
                                    if (baseOffset < 0)
                                    {
                                        // offset at src will not make all src tensors to fit in src's buffer
                                        // bailout 
                                        optimize = false;
                                        break;
                                    }
                                    srcFound = true;
                                    continue; // next src
                                }
                                if (srcFound)
                                {
                                    // we have found a pre-assigned src, check that buffer ID and offsets match
                                    NGRAPH_CHECK(bufferId >= 0, "Cannot re-use buffer negative buffer ID");
                                    attr = getBufferId(defOp);
                                    if (attr)
                                    {

                                        if (bufferId != attr.getInt() || 
                                            getBufferOffset(defOp).getInt() != (baseOffset + offsetDeltas[i]))
                                        {
                                            // incompatible buffer, bail out
                                            optimize = false;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        // we should have a valid buffer ID and offset at this point
                        NGRAPH_CHECK(optimize || (bufferId == -1 && baseOffset == 0));
                        if (optimize)
                        {  
                            // TODO: Write annotations now. No need to check if we are over-writing since they should all match. 

                        }
                        

                    }
                     
                        

                        
                    }
                }

            }

            // update liveness info

            for (auto dit : op->getResults())
            {
                liveness.kill(dit);
            }
            for (auto uit : op->getOperands())
            {
                liveness.setLive(uit);
            }
        }
        #if 0
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
        #endif 
    }

    bool MemoryOptimizationPass::isInputOrOutputValue(mlir::Value* value)
    {
        // do we have a buffer id attached to this value
        auto defOp = value->getDefiningOp();
        // If no defining op, then this is a block arg, skip operand
        if (!defOp)
        {
            return true;
        }

        // If the defined value is an output of the sub-graph, cannot do it in place
        for (auto& use : value->getUses())
        {
            auto useOp = use.getOwner();
            if (isa<NGReturnOp>(useOp))
            {
                return true;
            }
        }
        return false;
    }
    // TODO Change this to use interfaces. 
    bool MemoryOptimizationPass::isSafeInPlace(mlir::Operation* op)
    {
        auto it = m_inplaceOps.find(op->getName().getStringRef().str());
        return it != m_inplaceOps.end() ? it->second : false;
    }


    AliasRelation::AliasRelation()
    {
        for (auto i = 0; i < m_reachability.size(); i++)
        {
            auto& v = m_reachability[i];
            std::fill(v.begin(), v.end(), 0);
        }
        m_maxIdx = 0;
        m_needsTransitiveClosure = false;
    }
    bool AliasRelation::canAlias(Value* a, Value* b)
    {
        NGRAPH_CHECK(m_needsTransitiveClosure == false, "Relationship needs transitive closure");
        auto a_it = m_valueToIdx.find(a);
        auto b_it = m_valueToIdx.find(b);
        if (a_it == m_valueToIdx.end() || b_it == m_valueToIdx.end())
        {
            // at least one value doesn't exist in the cannot-alias relationship
            return true;
        }
        auto a_idx = a_it->second;
        auto b_idx = b_it->second;
        return (!m_reachability[a_idx][b_idx] && m_reachability[b_idx][a_idx]);
    }
    void AliasRelation::insertNoAlias(Value* a, Value* b)
    {
        m_needsTransitiveClosure = true;
        int rowsToAdd = 0;
        unsigned a_idx, b_idx;
        auto it = m_valueToIdx.find(a);
        if (it == m_valueToIdx.end())
        {
            rowsToAdd++;
            a_idx = m_maxIdx++;
            m_valueToIdx[a] = a_idx;
        }
        else
        {
            a_idx = it->second;
        }

        it = m_valueToIdx.find(b);
        if (it == m_valueToIdx.end())
        {
            rowsToAdd++;
            b_idx = m_maxIdx++;
            m_valueToIdx[b] = b_idx;
        }
        else
        {
            b_idx = it->second;
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
            m_reachability.push_back(Row(m_maxIdx /* size */, 0));
            if (rowsToAdd > 1)
            {
                m_reachability.push_back(Row(m_maxIdx /* size */, 0));
            }
        }

        m_reachability[a_idx][b_idx] = 1;
        m_reachability[b_idx][a_idx] = 1;

#ifdef NGRAPH_DEBUG_ENABLE
        checkInvariance();
#endif
        computeTransitiveClosure();
    }

    void AliasRelation::computeTransitiveClosure()
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

#ifdef NGRAPH_DEBUG_ENABLE
    void AliasRelation::checkInvariance() 
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
#endif

    void Liveness::reset()
    {
        m_valueToIdx.clear();
        m_liveness.clear();
        m_maxIdx = 0;
    }

    void Liveness::getLiveValues(llvm::SmallVectorImpl<Value*>& values)
    {
        for (auto& entry : m_valueToIdx)
        {
            if (m_liveness[entry.second])
            {
                values.push_back(entry.first);
            }
        }
    }

    bool Liveness::isLive(Value* v)
    {
        auto it = m_valueToIdx.find(v);
        if (it == m_valueToIdx.end())
        {
            return false;
        }
        return m_liveness[it->second];
    }

    void Liveness::setLive(Value* v)
    {
        auto it = m_valueToIdx.find(v);
        if (it == m_valueToIdx.end())
        {
            m_valueToIdx[v] = m_maxIdx++;
            NGRAPH_CHECK(m_liveness.size() == m_maxIdx);
            m_liveness.push_back(true);
        }
        else
        {
            m_liveness[it->second] = true;            
        }
    }

    void Liveness::kill(Value* v)
    {
        auto it = m_valueToIdx.find(v);
        if (it == m_valueToIdx.end())
        {
            // already dead
            return;
        }
        m_liveness[it->second] = false;
    }
}




namespace mlir
{
    std::unique_ptr<Pass> createMemoryOptimizationPass()
    {
        return std::make_unique<MemoryOptimizationPass>();
    }
} // namespace mlir
