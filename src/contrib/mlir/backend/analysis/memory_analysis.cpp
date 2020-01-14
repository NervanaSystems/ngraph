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

#include "memory_analysis.hpp"
#include "contrib/mlir/core/compiler.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/type.hpp"

#include <llvm/ADT/BitVector.h>
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

static llvm::cl::opt<bool> clEnableNgInPlaceMemory(
    "ngraph-memory-opt",
    llvm::cl::init(true),
    llvm::cl::desc("Enable ngraph dialect in-place memory optimization pass"));

static llvm::cl::opt<bool>
    clEnableNgInPlaceConcat("ngraph-memory-opt-concat",
                            llvm::cl::init(true),
                            llvm::cl::desc("Enable inplace concat optimization"));

static llvm::cl::opt<bool>
    clEnableNgInPlaceEltWise("ngraph-memory-opt-eltwise",
                             llvm::cl::init(true),
                             llvm::cl::desc("Enable inplace element wise optimization"));

// anonymous namespace
// no need to expose any of the following outside of this file
namespace
{
    using namespace ngraph::runtime;
    using namespace ngraph::runtime::ngmlir;
    using namespace mlir;

    // A helper data-structure to track cannot alias relationship between
    // tensor syms. If NoAlias[T] contains S, then T and S cannot alias.
    // The relationship is an equivalence (transitive, symmetric, reflexive)
    // Initially each sym is put in its own equivalence class (set).
    // If two syms a and b are found to be non-alias (equivalent),
    // their equivalence classes are unioned
    class AliasRelation
    {
    public:
        /// Initialize the relationship for a number of syms
        void init(std::unordered_set<Value*>& symbols);
        /// Checks if values a and b can alias
        bool canAlias(Value* a, Value* b);
        void insertNoAlias(Value* a, Value* b);

    private:
        using BV = llvm::BitVector;
        std::unordered_map<Value*, unsigned> m_valueToIdx;
        std::unordered_map<unsigned, Value*> m_idxToValue;
        std::unordered_map<Value*, BV*> m_valueToSet;
        SmallVector<BV, 10> m_sets;
    };

    // Simple single basic block liveness analysis
    // TODO: Replace with MLIR's liveness analysis
    class LivenessAnalysis
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

    // Memory Assignment analysis
    // Tries to find operations that can be done in place where applicable
    // by assigning a virtual buffer ID to values.
    // The buffer assignment is used later in affine lowering pass to create
    // or re-use memrefs
    class MemoryAssignment
    {
    public:
        MemoryAssignment(MemoryAnalysis* memAnalysis)
            : m_memAnalysis(memAnalysis)
        {
            m_inplaceOps = {
#define MLIR_OP(OP, INPLACE) {OP::getOperationName().str(), INPLACE},
#include "contrib/mlir/backend/pass/op_lowerers.inc"
            };
            m_bufferId = 0;
        }
        void run(ModuleOp* module);

    private:
        void processDestructiveInPlace(mlir::Operation* op);
        void processConcat(mlir::Operation* op);
        bool isSafeInPlace(mlir::Operation* op);
        bool isInputOrOutputValue(mlir::Value* value);
        LivenessAnalysis m_liveness;
        AliasRelation m_aliasRelation;
        std::unordered_map<std::string, bool> m_inplaceOps;
        int m_bufferId;
        MemoryAnalysis* m_memAnalysis;
    };

    // helpers
    // Determines the buffer size a value needs based on its type
    // offset is where that value should start in the buffer
    static unsigned getBufferSizeForOperand(mlir::Value* value, int offset);

    // Go backwards over instructions
    //
    // Re-use buffers if none of the dst/srcs are input/output of the sub-graph
    //
    // For destructive in-place ops (elt-wise):
    //      - Find first src where it is last use (src is dead).
    //        If all srcs are last-use, then pick one with lower number of uses.
    //        If no src is found, bail out.
    //      - If dst has pre-assigned buffer/offset, then copy them to src.
    //        If not, assign new buffer to both dst and src.
    //      - Mark all live syms at this point to not alias src
    //
    // For non-Destructive in-place ops:
    //      Concat:
    //          - Reuse buffer if
    //              - Concat axis is most-significant non-one axis, and
    //              - all operands can alias dest.
    //          - If dst has an assignment, copy it over to srcs as long as
    //          there is no conflicting src pre-assignment
    //          - If dst has no assignment, and all srcs have no assignment,
    //          assign new buffer to dst and srcs
    //
    //      Slice: TBD
    //      Reshape: TBD
    //
    // Update liveness info
    void MemoryAssignment::run(ModuleOp* module)
    {
        if (!clEnableNgInPlaceMemory)
        {
            // Optimization disabled
            return;
        }
        SmallVector<FuncOp, 2> funcOps(module->getOps<FuncOp>());

        if (funcOps.size() > 1 || funcOps.empty())
        {
            // single func for now
            return;
        }
        auto f = funcOps.back();
        auto& blocks = f.getBlocks();
        if (blocks.size() != 1)
        {
            // single block func for now
            return;
        }
        auto& block = *(blocks.begin());

        // count number of syms in the code and initialize alias relationship
        std::unordered_set<Value*> syms;

        for (auto it = block.begin(); it != block.end(); it++)
        {
            Operation* op = &(*it);
            for (auto it : op->getResults())
            {
                Value* v = it;
                if (syms.find(v) == syms.end())
                {
                    syms.insert(v);
                }
            }
            for (auto it : op->getOperands())
            {
                Value* v = it;
                if (syms.find(v) == syms.end())
                {
                    syms.insert(v);
                }
            }
        }
        m_aliasRelation.init(syms);
        // scan instructions backwards
        for (auto it = block.rbegin(); it != block.rend(); it++)
        {
            Operation* op = &(*it);

            if (isSafeInPlace(op))
            {
                // TODO: replace with Op Interface check
                if (dyn_cast<NGConcatOp>(op))
                {
                    if (clEnableNgInPlaceConcat)
                        processConcat(op);
                }
                else
                {
                    if (clEnableNgInPlaceEltWise)
                        processDestructiveInPlace(op);
                }
            }
            // update liveness info
            for (auto dit : op->getResults())
            {
                m_liveness.kill(dit);
            }
            for (auto uit : op->getOperands())
            {
                m_liveness.setLive(uit);
            }
        }
    }

    void MemoryAssignment::processConcat(mlir::Operation* op)
    {
        auto concat = cast<mlir::NGConcatOp>(op);
        {
            // concat on the highest non-one axis
            auto concatAxis = concat.concatenation_axis();
            auto result = concat.getResult();
            auto shape = (result->getType().cast<NGTensorType>()).getShape();
            std::vector<int> opndOffsets;
            BufferInfo bufferInfo;
            int bufferId = -1, baseOffset = 0;
            unsigned bufferSize = 0;

            if (isInputOrOutputValue(op->getResult(0)))
            {
                // dst is output, bail out
                return;
            };

            for (auto i = 0; i < shape.size(); i++)
            {
                if (i == concatAxis)
                {
                    break;
                }
                if (shape[i] != 1)
                {
                    return;
                }
            }
            // check that all operands and dst can alias
            // and that none is input or output
            for (auto opnd : op->getOperands())
            {
                if (!m_aliasRelation.canAlias(result, opnd) || isInputOrOutputValue(opnd))
                {
                    return;
                }
            }
            // calculate relative offsets in the output buffer
            int opndOffset = 0;
            for (auto i = 0; i < op->getNumOperands(); i++)
            {
                if (i == 0)
                {
                    opndOffsets.push_back(0);
                }
                else
                {
                    auto opnd = op->getOperand(i - 1);
                    auto tensorType = opnd->getType().cast<NGTensorType>();
                    opndOffset += tensorType.getNumElements();
                    opndOffsets.push_back(opndOffset);
                }
            }
            // check for consistent pre-existing buffer assignments
            bufferInfo = m_memAnalysis->getBufferInfo(op);
            // if dest has an assignment
            if (bufferInfo.isValid())
            {
                // set buffer ID and base offset to that of dest's
                bufferId = bufferInfo.m_bufferId;
                baseOffset = bufferInfo.m_offset;
                // check if we can re-use it for all src operands
                int bufferOffset = 0;
                for (auto i = 0; i < op->getNumOperands(); i++)
                {
                    auto opnd = op->getOperand(i);
                    auto defOp = opnd->getDefiningOp();
                    NGRAPH_CHECK(defOp != nullptr, "Defining operation expected");
                    // calculate expected absolute offset in the buffer
                    bufferOffset = baseOffset + opndOffsets[i];

                    bufferInfo = m_memAnalysis->getBufferInfo(defOp);
                    if (bufferInfo.isValid())
                    {
                        if (bufferInfo.m_bufferId != bufferId ||
                            bufferInfo.m_offset != bufferOffset)
                        {
                            // buffer ID or offset mismatch, bailout
                            return;
                        }
                    }
                }
            }
            else
            {
                // dst has no buffer assignment
                // TODO:
                // We can re-use an existing assignment of a src operand if
                // Every other src either:
                //    a. has a matching pre-assigned buffer ID and offset, or
                //    b. is unassigned a buffer/offset, and the computed offset is valid
                //       (non-negative), and no other live tensor aliases the chunk
                //       of the buffer we want to assign.
                //       To achieve this, we need to track buffer->{tensor,offset,size} and
                //       perform the check
                //
                // Example:
                // V1   = Concat    S0 (?), S1{0,16}, S2 (?)
                // R0   = ...
                // R2   = ...
                // V2   = Concat    R0{0, 0}, S1 {0,16}, R2{0,32}
                //
                // For the first concat, we could use the assignment of S1 (from second concat)
                // to define assignments for S0 and S2, and since R0, R2 are dead, no live tensors
                // alias into the buffer, and the assignment is valid.
                //
                // On the other hand, the following is invalid
                // Example:
                // R0   = ...
                // V1   = Concat    S0(?), S1(0,16), S2(?)
                // R2   = ...
                // V2   = Concat    R0, S1{0,16}, R2
                // Reusing assignment of S1 in the first concat will cause S0 and R0 to alias.
                // And since R0 is alive the write to R0 will overwrite S0.
                // For now, assign only if all srcs have no prior assignments
                for (auto opnd : op->getOperands())
                {
                    if (m_memAnalysis->getBufferInfo(opnd->getDefiningOp()).isValid())
                    {
                        return;
                    }
                }
            }
            // We didn't find any pre-existing buffer assignment, create a new buffer
            if (bufferId == -1)
            {
                bufferId = m_bufferId++;
                baseOffset = 0;
            }

            // adjust the buffer size based on this instruction
            // max size is determined from dst offset and type
            bufferSize = getBufferSizeForOperand(op->getResult(0), baseOffset);
            m_memAnalysis->setBufferSize(bufferId, bufferSize);

            // Update analysis map. No need to check if we are over-writing previous entries
            // since they should all match.
            m_memAnalysis->setBufferInfo(op, {bufferId, baseOffset});
            for (auto i = 0; i < op->getNumOperands(); i++)
            {
                auto opnd = op->getOperand(i);
                auto defOp = opnd->getDefiningOp();
                NGRAPH_CHECK(defOp != nullptr, "Defining operation expected");
                auto opndOffset = baseOffset + opndOffsets[i];
                m_memAnalysis->setBufferInfo(defOp, {bufferId, opndOffset});
            }
        }
    }

    void MemoryAssignment::processDestructiveInPlace(mlir::Operation* op)
    {
        NGRAPH_CHECK(op->getNumResults() == 1, "Destructive in-place with multi-def ?");
        Value* use = nullptr;
        int useCount = -1;

        if (isInputOrOutputValue(op->getResult(0)))
        {
            // dst is output, bail out
            return;
        };
        // pick a dead operand that is not an input or output with the least number of uses
        for (auto opnd : op->getOperands())
        {
            if (!m_liveness.isLive(opnd) && !isInputOrOutputValue(opnd))
            {
                int uses = 0;
                for (auto& i : opnd->getUses())
                {
                    uses++;
                }
                if (useCount == -1 || uses < useCount)
                {
                    use = opnd;
                    useCount = uses;
                }
            }
        }
        if (!use)
        {
            return;
        }
        // assign new buffer or copy buffer info from dst
        auto bufferInfo = m_memAnalysis->getBufferInfo(op);
        if (!bufferInfo.isValid())
        {
            // attach a new buffer id, and 0 offset on obth src and result
            bufferInfo = {m_bufferId++, 0};
            m_memAnalysis->setBufferInfo(op, bufferInfo);
            m_memAnalysis->setBufferInfo(use->getDefiningOp(), bufferInfo);
        }
        else
        {
            // copy result buffer id and offset to src
            m_memAnalysis->setBufferInfo(use->getDefiningOp(), bufferInfo);
        }
        auto bufferSize = 0;
        bufferSize = getBufferSizeForOperand(op->getResult(0), bufferInfo.m_offset);
        m_memAnalysis->setBufferSize(bufferInfo.m_bufferId, bufferSize);
        // update aliasing info
        // use value cannot alias any live value
        SmallVector<Value*, 10> liveValues;
        m_liveness.getLiveValues(liveValues);
        for (auto& value : liveValues)
        {
            m_aliasRelation.insertNoAlias(use, value);
        }
    }
    bool MemoryAssignment::isInputOrOutputValue(mlir::Value* value)
    {
        auto defOp = value->getDefiningOp();
        // If no defining op, then this is a block arg, skip operand
        //
        // TODO: This check is assuming single BB function, improve to handle control-flow.
        // In which case, we have to track block args to all pred branches that feed them,
        // all the way up to the initial def, if any, or entry block arg. This is preferably
        // done as a pre-pass to capture all inputs/output values.
        if (!defOp)
        {
            return true;
        }
        // If the defined value is an output of the sub-graph, cannot do it in place
        //
        // TODO: Improve to support control flow. Track value use-chain along branches/block-args,
        // if we hit a use in a return, it is an output value.
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
    bool MemoryAssignment::isSafeInPlace(mlir::Operation* op)
    {
        auto it = m_inplaceOps.find(op->getName().getStringRef().str());

        return it != m_inplaceOps.end() ? it->second : false;
    }

    void AliasRelation::init(std::unordered_set<Value*>& symbols)
    {
        unsigned numSyms = symbols.size();
        m_sets.resize(numSyms);
        for (auto& bv : m_sets)
        {
            bv.resize(numSyms);
        }
        // populate id->value and value->id maps
        unsigned i = 0;
        for (auto v : symbols)
        {
            m_idxToValue[i] = v;
            m_valueToIdx[v] = i;
            m_valueToSet[v] = &m_sets[i];
            // set bit for that value
            m_sets[i].set(i);
            i++;
        }
    }

    bool AliasRelation::canAlias(Value* a, Value* b)
    {
        // check if a and b are in the same set
        return m_valueToSet[a] != m_valueToSet[b];
    }

    void AliasRelation::insertNoAlias(Value* a, Value* b)
    {
        // union the two sets that a and b belong to
        // update the maps accordingly
        if (!canAlias(a, b))
        {
            // nothing to do
            return;
        }
        // union the two sets of a and b
        BV* aSet = m_valueToSet[a];
        BV* bSet = m_valueToSet[b];
        BV uSet = (*aSet);
        uSet |= (*bSet);
        // replace aSet with union
        auto pSet = m_valueToSet[a];
        *pSet = uSet;
        // update value to set maps
        for (auto it = pSet->set_bits_begin(); it != pSet->set_bits_end(); it++)
        {
            unsigned id = *it;
            auto value = m_idxToValue[id];
            m_valueToSet[value] = pSet;
        }
    }

    void LivenessAnalysis::reset()
    {
        m_valueToIdx.clear();
        m_liveness.clear();
        m_maxIdx = 0;
    }

    void LivenessAnalysis::getLiveValues(llvm::SmallVectorImpl<Value*>& values)
    {
        for (auto& entry : m_valueToIdx)
        {
            if (m_liveness[entry.second])
            {
                values.push_back(entry.first);
            }
        }
    }

    bool LivenessAnalysis::isLive(Value* v)
    {
        auto it = m_valueToIdx.find(v);
        if (it == m_valueToIdx.end())
        {
            return false;
        }
        return m_liveness[it->second];
    }

    void LivenessAnalysis::setLive(Value* v)
    {
        auto it = m_valueToIdx.find(v);
        if (it == m_valueToIdx.end())
        {
            m_valueToIdx[v] = m_maxIdx++;
            m_liveness.push_back(true);
            NGRAPH_CHECK(m_liveness.size() == m_maxIdx);
        }
        else
        {
            m_liveness[it->second] = true;
        }
    }

    void LivenessAnalysis::kill(Value* v)
    {
        auto it = m_valueToIdx.find(v);
        if (it == m_valueToIdx.end())
        {
            // already dead
            return;
        }
        m_liveness[it->second] = false;
    }
    // helpers
    unsigned getBufferSizeForOperand(mlir::Value* value, int offset)
    {
        auto tensorType = value->getType().dyn_cast<NGTensorType>();
        NGRAPH_CHECK(tensorType, "Invalid type to find buffer size for");

        unsigned bufferSize = offset * std::ceil(tensorType.getElementBitWidth() / 8);
        bufferSize += tensorType.getSizeInBytes();

        return bufferSize;
    }
}

namespace mlir
{
    MemoryAnalysis::MemoryAnalysis(Operation* op)
    {
        MemoryAssignment memoryAssignment(this);
        auto moduleOp = dyn_cast<ModuleOp>(op);
        NGRAPH_CHECK(moduleOp != nullptr, "Expecting FuncOp for anaylsis");
        memoryAssignment.run(&moduleOp);
    }
} // namespace mlir
