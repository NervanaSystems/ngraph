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

#include "compiler.hpp"
#include "dialect/ops.hpp"
#include "dialect/type.hpp"
#include "ngraph/assertion.hpp"

#include <llvm/ADT/DenseSet.h>
#include <mlir/EDSC/Builders.h>
#include <mlir/EDSC/Helpers.h>
#include <mlir/EDSC/Intrinsics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <map>

// anonymous namespace
// no need to expose any of the following outside of this file
namespace
{
    using namespace mlir;
    using namespace mlir::edsc;
    using namespace ngraph::runtime;

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
        void initConverters(OwningRewritePatternList& patterns, MLIRContext* mlirContext) override
        {
            RewriteListBuilder<NGAddOpConversion, NGDotOpConversion, NGReturnOpConversion>::build(
                patterns, mlirContext, m_pass);
        }

    private:
        DialectLoweringPass& m_pass;
        llvm::BumpPtrAllocator allocator;
    };

    /// Dialect Lowering Pass to affine ops
    class DialectLoweringPass : public ModulePass<DialectLoweringPass>
    {
    public:
        DialectLoweringPass(ngmlir::MLIRCompiler& compiler)
            : m_dialectLowerer(*this)
            , m_compiler(compiler)
        {
        }
        void runOnModule() override;
        SmallVector<Value*, 4> buildOutputDefs(Operation* op, PatternRewriter& rewriter);

    private:
        mlir::Function* getCallDecl(StringRef name,
                                    ArrayRef<Type> args,
                                    ArrayRef<Type> output,
                                    PatternRewriter& rewriter);
        void findOutputValues();
        void processFakeInstrs();
        Value* insertMemMgrDef(PatternRewriter* rewriter = nullptr);

    private:
        DialectLowerer m_dialectLowerer;
        // Value holding mem manager passed pointer
        SmallVector<Value*, 4> m_memMgrDefs;

        // list of results values to add to func signature
        SmallVector<Value*, 4> m_loweredOutputValues;
        ngmlir::MLIRCompiler& m_compiler;
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

        processFakeInstrs();
    }

    void DialectLoweringPass::findOutputValues()
    {
        // get original function
        auto f = getModule().getNamedFunction("main");
        SmallVector<Value*, 4> outputList;
        unsigned outputCount = 0;

        // we find out output values by looking at returned values
        // any return should return all outputs of the subgraph
        f->walk<NGReturnOp>([this, &outputCount](NGReturnOp ret) {
            for (unsigned i = 0; i < ret.getNumOperands(); i++)
            {
                auto outputValue = ret.getOperand(i);
                auto op = outputValue->getDefiningOp();
                op->setAttr("graphOutputIdx",
                            mlir::IntegerAttr::get(IntegerType::get(8, op->getContext()), i));
            }
            NGRAPH_CHECK(outputCount == 0 || outputCount == ret.getNumOperands(),
                         "Inconsistent returns in function");
            outputCount = ret.getNumOperands();
        });
        // will be populated with lowered output values later
        m_loweredOutputValues.resize(outputCount, nullptr);
    }

    /// Inserts a fake def for Mem Mgr pointer at converted func start
    Value* DialectLoweringPass::insertMemMgrDef(PatternRewriter* rewriter)
    {
        // it would be nice to insert one fake def at the start of the new func
        // however, due to how DialectConversion framework works, new func is only
        // materialized after conversion is done (rewriter->getFunction, or even rewriter->getInsertionBlock()->getFunction()
        // will give you the original func). This makes it very convoluted to insert instructions at entry block.
        auto op = rewriter->create<NGFakeInputOp>(rewriter->getUnknownLoc(),
                                                  IndexType::get(getModule().getContext()));
        // will be fixed later to read passed arg instead.
        m_memMgrDefs.push_back(op.getResult());
        return op.getResult();
    }

    SmallVector<Value*, 4> DialectLoweringPass::buildOutputDefs(Operation* op,
                                                                PatternRewriter& rewriter)
    {
        SmallVector<Value*, 4> newResults;
        for (auto origResult : op->getResults())
        {
            // create output def if this operation produces any sub-graph outputs
            if (IntegerAttr attr = op->getAttrOfType<IntegerAttr>("graphOutputIdx"))
            {
                unsigned argId = (int)attr.getInt();
                auto fakeOp = rewriter.create<NGFakeInputOp>(
                    op->getLoc(),
                    m_dialectLowerer.convertType(
                        origResult->getType()) /* convert to lowered type */
                    );
                // Fake instrution is short-lived. Verify here.
                fakeOp.verify();
                auto newResult = fakeOp.getResult();
                newResults.push_back(newResult);
                m_loweredOutputValues[argId] = newResult;
            }
            else
            {
                auto tensorType = origResult->getType().cast<NGTensorType>();
                auto callBackFunc = getCallDecl("__mlir_allocate",
                                                {rewriter.getIndexType(), rewriter.getIndexType()},
                                                {m_dialectLowerer.convertType(tensorType)},
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
            NGRAPH_CHECK(isa<NGFakeInputOp>(op), "output value not defined by fake output?");
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
                                                     PatternRewriter& rewriter)
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
    Type DialectLowerer::convertType(Type type)
    {
        // We may need to refactor this code to a external utility if type conversion is needed
        // outside of the lowering context since DialectLowerer is private.

        if (auto tensor_type = type.dyn_cast<NGTensorType>())
        {
            // Convert NGTensorType to Std MemRefType directly instead of going to Std TensorType.
            // This may change in the future.
            return MemRefType::get(tensor_type.getShape(),
                                   convertType(tensor_type.getElementType()),
                                   {/* no map used */},
                                   0);
        }
        if (auto float_type = type.dyn_cast<NGFloatType>())
        {
            // Float types are already std type.
            return float_type;
        }
        if (auto int_type = type.dyn_cast<NGIntegerType>())
        {
            return mlir::IntegerType::get(int_type.getWidth(), int_type.getContext());
        }
        if (auto bool_type = type.dyn_cast<NGBoolType>())
        {
            return mlir::IntegerType::get(1 /* width */, bool_type.getContext());
        }

        NGRAPH_CHECK(false, "Unsupported type to lower");
        return type;
    }

#define REWRITER(OP)                                                                               \
    void OP##Conversion::rewrite(                                                                  \
        Operation* op, ArrayRef<Value*> operands, PatternRewriter& rewriter) const

    // ADD
    REWRITER(NGAddOp)

    {
        auto add = cast<NGAddOp>(op);
        auto loc = add.getLoc();

        auto result = m_pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result->getType().isa<MemRefType>());
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
        // clang-format off
        LoopNestBuilder(pivs, lbs, ubs, steps)( 
            // single stmt body
            [&] {
                    iRes(ivs) = iLHS(ivs) + iRHS(ivs);
                });
        // clang-format on
        rewriter.replaceOp(op, {result});
    }

    REWRITER(NGDotOp)
    {
        auto dot = cast<NGDotOp>(op);
        auto loc = dot.getLoc();

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value* lhs = operands[0];
        Value* rhs = operands[1];
        Value* result = m_pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs && rhs && result, "Unexpected null values in DotOp");

        auto result_ty = result->getType().dyn_cast<MemRefType>();
        auto lhs_ty = lhs->getType().dyn_cast<MemRefType>();
        auto rhs_ty = rhs->getType().dyn_cast<MemRefType>();
        NGRAPH_CHECK(result_ty, "Unexpected non-memref result type");
        NGRAPH_CHECK(lhs_ty, "Unexpected non-memref LHS type");
        NGRAPH_CHECK(rhs_ty, "Unexpected non-memref RHS type");

        Type elem_ty = result_ty.getElementType();
        NGRAPH_CHECK(elem_ty == lhs_ty.getElementType() && elem_ty == rhs_ty.getElementType(),
                     "Types mismatch in DotOp");

        // Create the following loop nest for matmul operation:
        //   for(n, N, 1)
        //     for(m, M, 1)
        //       for(k, K, 1)
        //         res[n, k] += lhs[n, m] * rhs[m, k]
        // TODO (dcab): We currently generate a super naive loop nest. Improve loop nest layout.

        MemRefView v_res(result), v_lhs(lhs), v_rhs(rhs);

        NGRAPH_CHECK(v_lhs.rank() == 2 && v_rhs.rank() == 2 && v_res.rank() == 2,
                     "Dot operation is only supported for 2D tensors");

        // Create induction variables, lower bounds, upper bounds and steps of the loop nest.
        // It's important to note that MemRefView priovides lb/ub/step info is "reverse order",
        // i.e., fastest varying dimension is the last one, slowest varying dimention is the first
        // one.
        IndexHandle n, m, k;
        unsigned n_dim = v_lhs.fastestVarying() - 1;
        unsigned m_dim = v_rhs.fastestVarying();
        unsigned k_dim = v_rhs.fastestVarying();
        IndexHandle n_lb(v_lhs.lb(n_dim)), m_lb(v_lhs.lb(m_dim)), k_lb(v_rhs.lb(k_dim));
        IndexHandle n_ub(v_lhs.ub(n_dim)), m_ub(v_lhs.ub(m_dim)), k_ub(v_rhs.ub(k_dim));
        int64_t n_step = v_lhs.step(n_dim), m_step = v_lhs.step(m_dim), k_step = v_rhs.step(k_dim);

        // Constants, indexed values and indexes to be used inside the loop nest.
        IndexedValue i_res(result), i_lhs(lhs), i_rhs(rhs);
        ValueHandle zero_init(rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elem_ty)));

        LoopBuilder(&n, n_lb, n_ub, n_step)([&] {
            LoopBuilder(&k, k_lb, k_ub, k_step)([&] {
                i_res(n, k) = zero_init;
                LoopBuilder(&m, m_lb, m_ub, m_step)(
                    [&] { i_res(n, k) += i_lhs(n, m) * i_rhs(m, k); });
            });
        });

        rewriter.replaceOp(op, {result});
    }

    REWRITER(NGReturnOp) { rewriter.replaceOpWithNewOp<ReturnOp>(op); }
#undef REWRITER
}

namespace mlir
{
    Pass* createDialectLoweringPass(ngraph::runtime::ngmlir::MLIRCompiler* compiler)
    {
        return new DialectLoweringPass(*compiler);
    }
}
