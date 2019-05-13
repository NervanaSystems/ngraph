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

#include "lowerer.hpp"
#include <map>
#include "compiler.hpp"
#include "llvm/ADT/DenseSet.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Helpers.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ngraph/assertion.hpp"
#include "dialect/ops.hpp"
#include "dialect/type.hpp"

using namespace ngraph::runtime::ngmlir;
// anonymous namespace
// no need to expose any of the following outside of this file
namespace
{
    using namespace mlir;
    using namespace mlir::edsc;
    using namespace ngraph::runtime::ngmlir;

    class DialectLoweringPass;
#include "op_lowerers.inc"

    /// Use Dialect Converson Framework
    class DialectLowerer : public DialectConversion
    {
    public:
        DialectLowerer(DialectLoweringPass& pass)
            : DialectConversion()
            , m_pass(pass)
        {
        }

        Type convertType(Type t) override;

    protected:
        // Initialize the list of converters.
        llvm::DenseSet<DialectOpConversion*> initConverters(MLIRContext* context) override
        {
            return ConversionListBuilder<NG_AddOpConversion,
                                         NG_MatmulBiasOpConversion,
                                         NG_ReturnOpConversion>::build(&allocator, context, m_pass);
        }

    private:
        DialectLoweringPass& m_pass;
        llvm::BumpPtrAllocator allocator;
    };

    /// Dialect Lowering Pass to affine ops
    class DialectLoweringPass : public ModulePass<DialectLoweringPass>
    {
    public:
        DialectLoweringPass(MLIRCompiler& compiler)
            : m_dialectLowerer(*this)
            , m_compiler(compiler)
        {
        }
        void runOnModule() override;
        std::map<Value*, unsigned>& getOutputValueMap() { return m_outputValueMap; };
        SmallVector<Value*, 4> buildOutputDefs(Operation* op, FuncBuilder& rewriter);

    private:
        mlir::Function* getCallDecl(StringRef name,
                                    ArrayRef<Type> args,
                                    ArrayRef<Type> output,
                                    FuncBuilder& rewriter);
        void findOutputValues();
        void processFakeInstrs();
        Value* insertMemMgrDef(FuncBuilder* rewriter = nullptr);

    private:
        DialectLowerer m_dialectLowerer;
        // Value holding mem manager passed pointer
        SmallVector<Value*, 4> m_memMgrDefs;
        // maps output ng dialect values to args pos
        std::map<Value*, unsigned> m_outputValueMap;
        // list of results values to add to func signature
        SmallVector<Value*, 4> m_loweredOutputValues;
        MLIRCompiler& m_compiler;
    };

    void DialectLoweringPass::runOnModule()
    {
        // capture output values by looking for the Return and grabbing the values
        // the order of the returned values matches the order of the lowered func signature for
        // results. This is used to find the arg_id that a defined value maps to if it is an output
        findOutputValues();

        if (failed(m_dialectLowerer.convert(&getModule())))
        {
            getModule().getContext()->emitError(mlir::UnknownLoc::get(getModule().getContext()),
                                                "Error lowering dialect\n");
            signalPassFailure();
        }
        if (std::getenv("NGRAPH_MLIR_DUMP_ALL") != nullptr)
        {
            getModule().dump();
        }
        processFakeInstrs();
        if (std::getenv("NGRAPH_MLIR_DUMP_ALL") != nullptr)
        {
            getModule().dump();
        }
    }

    void DialectLoweringPass::findOutputValues()
    {
        // get original function
        auto f = getModule().getNamedFunction("main");
        SmallVector<Value*, 4> outputList;
        unsigned outputCount = 0;

        // we find out output values by looking at returned values
        // any return should return all outputs of the subgraph
        f->walk<NG_ReturnOp>([this, &outputCount](NG_ReturnOp ret) {
            for (unsigned i = 0; i < ret.getNumOperands(); i++)
            {
                this->m_outputValueMap.insert(std::pair<Value*, unsigned>(ret.getOperand(i), i));
            }
            NGRAPH_ASSERT(outputCount == 0 || outputCount == ret.getNumOperands())
                << "Inconsistent returns in function";
            outputCount = ret.getNumOperands();
        });
        // will be populated with lowered output values later
        m_loweredOutputValues.resize(outputCount, nullptr);
    }

    /// Inserts a fake def for Mem Mgr pointer at converted func start
    Value* DialectLoweringPass::insertMemMgrDef(FuncBuilder* rewriter)
    {
        // it would be nice to insert one fake def at the start of the new func
        // however, due to how DialectConversion framework works, new func is only
        // materialized after conversion is done (rewriter->getFunction, or even rewriter->getInsertionBlock()->getFunction()
        // will give you the original func). This makes it very convoluted to insert instructions at entry block.
        auto op = rewriter->create<NG_FakeInput>(rewriter->getUnknownLoc(),
                                                 IndexType::get(getModule().getContext()));
        // will be fixed later to read passed arg instead.
        m_memMgrDefs.push_back(op.getResult());
        return op.getResult();
    }

    SmallVector<Value*, 4> DialectLoweringPass::buildOutputDefs(Operation* op,
                                                                FuncBuilder& rewriter)
    {
        auto& outputMap = getOutputValueMap();
        SmallVector<Value*, 4> newResults;
        for (auto origResult : op->getResults())
        {
            auto it = outputMap.find(origResult);
            // create output def if this operation produces any sub-graph outputs
            if (it != outputMap.end())
            {
                unsigned argId = (*it).second;
                auto newResult = rewriter
                                     .create<NG_FakeInput>(
                                         op->getLoc(),
                                         m_dialectLowerer.convertType(
                                             origResult->getType()) /* convert to lowered type */
                                         )
                                     .getResult();
                newResults.push_back(newResult);
                m_loweredOutputValues[argId] = newResult;
            }
            else
            {
                auto tensorType = origResult->getType().cast<NGTensorType>();
                auto callBackFunc = getCallDecl("__mlir_allocate",
                                                {rewriter.getIndexType(), rewriter.getIndexType()},
                                                {tensorType.toMemref()},
                                                rewriter);

                auto size = tensorType.getSizeInBytes();
                SmallVector<mlir::Value*, 4> args = {
                    insertMemMgrDef(&rewriter), /* pointer to mem manager */
                    rewriter.create<mlir::ConstantIndexOp>(rewriter.getUnknownLoc(),
                                                           size)}; /* size to allocate */
                auto newResult =
                    rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args)
                        .getResult(0);
                newResults.push_back(newResult);
            }
        }
        return newResults;
    }

    void DialectLoweringPass::processFakeInstrs()
    {
        auto context = getModule().getContext();
        auto f = getModule().getNamedFunction("main");
        mlir::Block* entryBlock = &*(f->begin());
        auto oldFuncType = f->getType();
        ArrayRef<mlir::Type> ipArgs = oldFuncType.getInputs();
        ArrayRef<mlir::Type> opArgs = oldFuncType.getResults();
        SmallVector<mlir::Type, 4> allArgs;

        // Move all args as inputs in new type
        for (auto type : ipArgs)
        {
            allArgs.push_back(type);
        }
        for (auto type : opArgs)
        {
            allArgs.push_back(type);
            // add new value for result
            entryBlock->addArgument(type);
        }
        // Mem Manager Ptr
        auto indexType = mlir::IndexType::get(context);
        allArgs.push_back(indexType);
        entryBlock->addArgument(indexType);
        // update type
        auto newFuncType = mlir::FunctionType::get(allArgs, {}, context);
        f->setType(newFuncType);

        // RAUW fake outputs with result values
        unsigned i = 0;
        for (auto value : m_loweredOutputValues)
        {
            auto op = value->getDefiningOp();
            NGRAPH_ASSERT(op->isa<NG_FakeInput>()) << "output value not defined by fake output?";
            value->replaceAllUsesWith(entryBlock->getArgument(oldFuncType.getNumInputs() + i));
            op->erase();
            i++;
        }
        for (auto v : m_memMgrDefs)
        {
            v->replaceAllUsesWith(entryBlock->getArgument(m_compiler.get_mem_mgr_arg_id(f)));
            v->getDefiningOp()->erase();
        }
    }

    mlir::Function* DialectLoweringPass::getCallDecl(StringRef name,
                                                     ArrayRef<Type> args,
                                                     ArrayRef<Type> output,
                                                     FuncBuilder& rewriter)
    {
        auto callBackFuncPtr = getModule().getNamedFunction(name);
        if (callBackFuncPtr == nullptr)
        {
            auto callBackType = rewriter.getFunctionType(args, output);
            auto callBackFunc =
                llvm::make_unique<mlir::Function>(rewriter.getUnknownLoc(), name, callBackType);
            callBackFuncPtr = callBackFunc.get();
            getModule().getFunctions().push_back(callBackFunc.release());
        }
        return callBackFuncPtr;
    }
    // NGDialect converters
    Type DialectLowerer::convertType(Type t)
    {
        if (auto tensor = t.dyn_cast<NGTensorType>())
        {
            return tensor.toMemref();
        }
        return t;
    }

    // ADD
    SmallVector<Value*, 4> NG_AddOpConversion::rewrite(Operation* op,
                                                       ArrayRef<Value*> operands,
                                                       FuncBuilder& rewriter) const
    {
        auto add = op->cast<NG_AddOp>();
        auto loc = add.getLoc();
        Value *origResult, *newResult;

        auto result = m_pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_ASSERT(result->getType().isa<MemRefType>());
        // Note that builder's current function is still the original function body.
        // use getBlock to get the new block instead.

        // get new operands
        Value* lhs = operands[0];
        Value* rhs = operands[1];

        ScopedContext scope(rewriter, loc);
        // Views
        MemRefView vRes(result), vLHS(lhs), vRHS(rhs);
        // Index Values
        IndexedValue iRes(result), iLHS(lhs), iRHS(rhs);
        // Bounds Index Handles
        auto lbs = vLHS.getLbs();
        auto ubs = vLHS.getUbs();
        // Loop induction vars
        auto ivs = IndexHandle::makeIndexHandles(vLHS.rank());
        auto pivs = IndexHandle::makeIndexHandlePointers(ivs);
        // Steps
        auto steps = vLHS.getSteps();

        LoopNestBuilder(pivs, lbs, ubs, steps)({// single stmt body
                                                iRes(ivs) = iLHS(ivs) + iRHS(ivs)});
        // return result memref
        return {result};
    }

    SmallVector<Value*, 4> NG_MatmulBiasOpConversion::rewrite(Operation* op,
                                                              ArrayRef<Value*> operands,
                                                              FuncBuilder& rewriter) const
    {
        auto matmul = op->cast<NG_MatmulBiasOp>();
        auto loc = matmul.getLoc();

        NGRAPH_ASSERT(!matmul.getBias() && operands.size() == 2)
            << "Bias is not supported yet in MatmulBias operation";

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value* lhs = operands[0];
        Value* rhs = operands[1];
        Value* result = m_pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_ASSERT(lhs && rhs && result) << "Unexpected null values in MatmulBiasOp";

        auto result_ty = result->getType().dyn_cast<MemRefType>();
        auto lhs_ty = lhs->getType().dyn_cast<MemRefType>();
        auto rhs_ty = rhs->getType().dyn_cast<MemRefType>();
        NGRAPH_ASSERT(result_ty) << "Unexpected non-memref result type";
        NGRAPH_ASSERT(lhs_ty) << "Unexpected non-memref LHS type";
        NGRAPH_ASSERT(rhs_ty) << "Unexpected non-memref RHS type";

        Type elem_ty = result_ty.getElementType();
        NGRAPH_ASSERT(elem_ty == lhs_ty.getElementType() && elem_ty == rhs_ty.getElementType())
            << "Types mismatch in MatmulBiasOp";

        // Create the following loop nest for matmul operation:
        //   for(n, N, 1)
        //     for(m, M, 1)
        //       for(k, K, 1)
        //         res[n, k] += lhs[n, m] * rhs[m, k]
        // TODO (dcab): We currently generate a super naive loop nest. Improve loop nest layout.

        MemRefView v_res(result), v_lhs(lhs), v_rhs(rhs);
        IndexedValue i_res(result), i_lhs(lhs), i_rhs(rhs);

        NGRAPH_ASSERT(v_lhs.rank() == 2 && v_rhs.rank() == 2 && v_res.rank() == 2)
            << "MatmulBias operation is only supported for 2D tensors";

        // Induction variables, lower bounds, upper bounds and steps of the loop nest.
        IndexHandle n, m, k;
        IndexHandle n_lb(v_lhs.lb(1)), m_lb(v_lhs.lb(0)), k_lb(v_rhs.lb(0));
        IndexHandle n_ub(v_lhs.ub(1)), m_ub(v_lhs.ub(0)), k_ub(v_rhs.ub(0));
        int64_t n_step = v_lhs.step(1), m_step = v_lhs.step(0), k_step = v_rhs.step(0);
        // TODO (dcab): Assert on dims

        // Constants, indexed values and indexes to be used inside the loop nest.
        IndexedValue ires(result), ilhs(lhs), irhs(rhs);
        ValueHandle zero_init(rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elem_ty)));

        // clang-format off
        LoopBuilder(&n, n_lb, n_ub, n_step)({
            LoopBuilder(&k, k_lb, k_ub, k_step)({
                i_res(n, k) = zero_init,
                LoopBuilder(&m, m_lb, m_ub, m_step)({
                    i_res(n, k) += i_lhs(n, m) * i_rhs(m, k)
                })
            }),
        });
        // clang-format on

        // Return result memref.
        return {result};
    }

    SmallVector<Value*, 4> NG_ReturnOpConversion::rewrite(Operation* op,
                                                          ArrayRef<Value*> operands,
                                                          FuncBuilder& rewriter) const
    {
        rewriter.create<ReturnOp>(op->getLoc());
        return {};
    }
}

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            Pass* createDialectLoweringPass(MLIRCompiler* compiler)
            {
                return new DialectLoweringPass(*compiler);
            }
        }
    }
}
