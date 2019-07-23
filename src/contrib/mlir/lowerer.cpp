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
    using namespace mlir::edsc::op;
    using namespace ngraph::runtime;
    using namespace ngraph::runtime::ngmlir;

    class DialectLoweringPass;

    /// Base class for nGraph operation conversions to affine/standard dialect. Provides
    /// conversion patterns with an access to the DialectLoweringPass which holds the state of the
    /// conversion.
    class NGraphOpLowering : public ConversionPattern
    {
    public:
        NGraphOpLowering(StringRef rootOpName, MLIRContext* context, DialectLoweringPass& pass)
            : ConversionPattern(rootOpName, /*benefit=*/1, context)
            , m_pass(pass){};

    protected:
        // Back-reference to the lowering pass which contains the lowering state, including the
        // nGraph type converter.
        DialectLoweringPass& m_pass;
    };

// Conversion classes declarations
#define MLIR_OP(OP)                                                                                \
    class OP##Conversion : public NGraphOpLowering                                                 \
    {                                                                                              \
    public:                                                                                        \
        explicit OP##Conversion(mlir::MLIRContext* context, DialectLoweringPass& pass)             \
            : NGraphOpLowering(mlir::OP::getOperationName(), context, pass)                        \
        {                                                                                          \
        }                                                                                          \
                                                                                                   \
        PatternMatchResult matchAndRewrite(Operation* op,                                          \
                                           ArrayRef<Value*> operands,                              \
                                           PatternRewriter& rewriter) const override;              \
    };

#include "op_lowerers.inc"

    // Helpers
    template <typename RedOp>
    void lowerIndexReduction(Operation* op,
                             ArrayRef<Value*> operands,
                             PatternRewriter& rewriter,
                             DialectLoweringPass& m_pass);

    template <typename OP>
    void lower_binary_elementwise(Operation* op,
                                  ArrayRef<Value*> operands,
                                  PatternRewriter& rewriter,
                                  DialectLoweringPass& m_pass);

    /// Conversion from types in the nGraph dialect to the Standard dialect.
    class NGraphTypeConverter : public TypeConverter
    {
    public:
        NGraphTypeConverter()
            : TypeConverter()
        {
        }

        Type convertType(Type t) override;
    };

    /// Dialect Lowering Pass to affine ops
    class DialectLoweringPass : public ModulePass<DialectLoweringPass>
    {
    public:
        DialectLoweringPass(ngmlir::MLIRCompiler& compiler)
            : m_compiler(compiler)
        {
        }

        void runOnModule() override;
        SmallVector<Value*, 4> buildOutputDefs(Operation* op, PatternRewriter& rewriter);
        Value* createTempTensor(Type type, PatternRewriter& rewriter);

        mlir::Function* getCallDecl(StringRef name,
                                    ArrayRef<Type> args,
                                    ArrayRef<Type> output,
                                    PatternRewriter& rewriter);

    private:
        /// Collect a set of patterns to convert from the nGraph dialect to Affine dialect.
        void populateNGraphToAffineConversionPatterns(OwningRewritePatternList& patterns);

        void findOutputValues();
        void processFakeInstrs();
        void insertNoAliasArgAttrs();
        Value* insertMemMgrDef(PatternRewriter* rewriter = nullptr);

    private:
        NGraphTypeConverter m_typeConverter;
        // Value holding mem manager passed pointer
        SmallVector<Value*, 4> m_memMgrDefs;

        // list of results values to add to func signature
        SmallVector<Value*, 4> m_loweredOutputValues;
        ngmlir::MLIRCompiler& m_compiler;
    };

    void DialectLoweringPass::runOnModule()
    {
        // Create type converter and initialize conversion patterns.
        NGraphTypeConverter converter;
        OwningRewritePatternList patterns;
        populateNGraphToAffineConversionPatterns(patterns);

        // Create target that defines legal ops for nGraph dialect to be lowered to.
        ConversionTarget target(getContext());
        // TODO: Remove NGFakeInputOp. We need to set NGFakeInputOp as legal op because we generate
        // it as part of the lowering to affine/standard.
        target.addLegalDialect<AffineOpsDialect, StandardOpsDialect>();
        target.addLegalOp<NGFakeInputOp>();

        // capture output values by looking for the Return and grabbing the values
        // the order of the returned values matches the order of the lowered func signature for
        // results. This is used to find the arg_id that a defined value maps to if it is an output
        findOutputValues();

        if (failed(applyConversionPatterns(getModule(), target, converter, std::move(patterns))))
        {
            emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering nGraph dialect\n");
            signalPassFailure();
        }

        processFakeInstrs();

        insertNoAliasArgAttrs();
    }

    void DialectLoweringPass::populateNGraphToAffineConversionPatterns(
        OwningRewritePatternList& patterns)
    {
#define MLIR_OP(OP) OP##Conversion,
#define MLIR_LAST_OP(OP) OP##Conversion
        RewriteListBuilder<
#include "op_lowerers.inc"
            >::build(patterns, &getContext(), *this);
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
        // TODO: This resize is making debugging obscure. When the container is not populated due
        // to a bug, null pointers are used by the consumer leading to a crash more difficult to
        // root-cause. We should try to change the current approach or introduce verification code.
        m_loweredOutputValues.resize(outputCount, nullptr);
    }

    /// Inserts a fake def for Mem Mgr pointer at converted func start
    Value* DialectLoweringPass::insertMemMgrDef(PatternRewriter* rewriter)
    {
        // it would be nice to insert one fake def at the start of the new func
        // however, due to how DialectConversion framework works, new func is only
        // materialized after conversion is done (rewriter->getFunction, or even
        // rewriter->getInsertionBlock()->getFunction() will give you the original func). This
        // makes it very convoluted to insert instructions at entry block.
        auto op = rewriter->create<NGFakeInputOp>(rewriter->getUnknownLoc(),
                                                  IndexType::get(&getContext()));
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
                    m_typeConverter.convertType(origResult->getType()) /* convert to lowered type */
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
                auto newResult =
                    createTempTensor(m_typeConverter.convertType(tensorType), rewriter);
                newResults.push_back(newResult);
            }
        }
        return newResults;
    }

    Value* DialectLoweringPass::createTempTensor(Type type, PatternRewriter& rewriter)
    {
        MemRefType memRefType = type.cast<MemRefType>();

        NGRAPH_CHECK(memRefType.hasStaticShape(), "Dynamic shapes are not supported");

        Value* alloc = rewriter.create<mlir::AllocOp>(rewriter.getUnknownLoc(), memRefType);

        // TODO:
        // Enable dynamic memref allocation via call-back to nGraph allocator
        // We should create a list of Values representing each dynamic dim
        // The values would be computed based on the shape of the input to the ng op we are lowering.
        // E.g. If lowering concat, Value for dynamic concat axis will be the sum of input dims.
        // The lowerer will generate code to compute the dims.
        // This is better be done via std.AllocOp but we need to make it hookable to nGraph allocator call-back.

        return alloc;
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

    /// Add llvm.noalias attribute to all the memref function arguments. We know that this is safe
    /// by nGraph op semantics.
    void DialectLoweringPass::insertNoAliasArgAttrs()
    {
        auto func = getModule().getNamedFunction("main");
        unsigned int argIdx = 0;
        for (auto* arg : func->getArguments())
        {
            if (arg->getType().isa<MemRefType>())
            {
                func->setArgAttr(argIdx, "llvm.noalias", BoolAttr::get(true, &getContext()));
            }

            ++argIdx;
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
    Type NGraphTypeConverter::convertType(Type type)
    {
        // We may need to refactor this code to a external utility if type conversion is needed
        // outside of the lowering context since NGraphTypeConverter is private.

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
    PatternMatchResult OP##Conversion::matchAndRewrite(                                            \
        Operation* op, ArrayRef<Value*> operands, PatternRewriter& rewriter) const

    // ADD
    REWRITER(NGAddOp)
    {
        lower_binary_elementwise<mlir::NGAddOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGSubOp)
    {
        lower_binary_elementwise<mlir::NGSubOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGMulOp)
    {
        lower_binary_elementwise<mlir::NGMulOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGDivOp)
    {
        lower_binary_elementwise<mlir::NGDivOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGGreaterOp)
    {
        lower_binary_elementwise<mlir::NGGreaterOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGLessOp)
    {
        lower_binary_elementwise<mlir::NGLessOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGMaxOp)
    {
        lower_binary_elementwise<mlir::NGMaxOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGMinOp)
    {
        lower_binary_elementwise<mlir::NGMinOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGArgMaxRedOp)
    {
        lowerIndexReduction<mlir::NGArgMaxRedOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    REWRITER(NGArgMinRedOp)
    {
        lowerIndexReduction<mlir::NGArgMinRedOp>(op, operands, rewriter, m_pass);
        return matchSuccess();
    }

    // Relu
    REWRITER(NGReluOp)
    {
        auto loc = cast<NGReluOp>(op).getLoc();

        auto result = m_pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result->getType().isa<MemRefType>());
        // Note that builder's current function is still the original function body.
        // use getBlock to get the new block instead.

        // get new operands
        Value* lhs = operands[0];

        ScopedContext scope(rewriter, loc);
        // Views
        MemRefView vRes(result), vLHS(lhs);
        // Index Values
        IndexedValue iRes(result), iLHS(lhs);
        // Bounds Index Handles
        auto lbs = vLHS.getLbs();
        auto ubs = vLHS.getUbs();
        // Loop induction vars
        auto ivs = IndexHandle::makeIndexHandles(vLHS.rank());
        auto pivs = IndexHandle::makeIndexHandlePointers(ivs);
        // Steps
        auto steps = vLHS.getSteps();

        NGRAPH_CHECK(lhs->getType().isa<MemRefType>());
        Type elemTy = lhs->getType().dyn_cast<MemRefType>().getElementType();
        NGRAPH_CHECK(!elemTy.isa<FloatType>(),
                     "NGReluOp with float element type should not be lowered until MLIR supports "
                     "lowering !std.CmpF");

        LoopNestBuilder(pivs, lbs, ubs, steps)([&] {
            ValueHandle val = iLHS(ivs);
            if (auto floatTy = elemTy.dyn_cast<FloatType>())
            {
                ValueHandle zero = intrinsics::constant_float(llvm::APFloat(0.0f), floatTy);
                iRes(ivs) = intrinsics::select(val > zero, val, zero);
            }
            else if (auto intTy = elemTy.dyn_cast<IntegerType>())
            {
                ValueHandle zero = intrinsics::constant_int(0, intTy.getWidth());
                iRes(ivs) = intrinsics::select(val > zero, val, zero);
            }
            else
            {
                NGRAPH_CHECK(false, "Unsupported type for Relu");
            }
        });

        rewriter.replaceOp(op, {result});
        return matchSuccess();
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

        // Constants and indexed values to be used inside the loop nest.
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

        return matchSuccess();
    }

    REWRITER(NGConcatOp)
    {
        auto concat = cast<NGConcatOp>(op);
        auto loc = concat.getLoc();
        ScopedContext scope(rewriter, loc);

        // Create Value for result, and extract type info.
        Value* result = m_pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in ConcatOp");

        // Create view to write into result.
        MemRefView vRes(result);
        auto rank = vRes.rank();

        // For each operand, generate a separate loop to copy into the target slice of "result".
        // We'll keep track of the slice offsets via concatenation_axis_pos.
        auto concatenationAxis = concat.concatenation_axis().getSExtValue();
        IndexHandle concatenationAxisPos(index_t(0));

        for (auto& operand : operands)
        {
            NGRAPH_CHECK(operand, "Unexpected null operand in ConcatOp");

            // Assuming rank = r, and the concatenation axis is A where A<r, we'll be creating
            // loops of this form:
            //
            //   for i_0 := 0 to operand.dims[0]:
            //    for i_1 := 0 to operand.dims[1]:
            //     ...
            //      for i_(r-2) := 0 to operand.dims[r-2]:
            //       for i_(r-1) := 0 to operand.dims[r-1]:
            //        result[i_0][i_1]...
            //              [i_(A-1)][i_A + concatenationAxisPos][i_(A+1)]...
            //              [i_(r-2)][i_(r-1)]
            //                  :=
            //        operand[i_0][i_1]...[i_(r-2)][i_(r-1)]
            MemRefView vOperand(operand);
            NGRAPH_CHECK(vOperand.rank() == rank, "Unexpected rank mismatch");

            llvm::SmallVector<ValueHandle, 5> indexVars;
            llvm::SmallVector<ValueHandle*, 5> indexVarPtrs;
            llvm::SmallVector<ValueHandle, 5> indexVarLbs;
            llvm::SmallVector<ValueHandle, 5> indexVarUbs;
            llvm::SmallVector<int64_t, 5> indexVarSteps;
            for (int i = 0; i < rank; i++)
            {
                indexVars.push_back(IndexHandle());
                indexVarPtrs.push_back(&(indexVars.back()));
                indexVarLbs.push_back(vOperand.lb(i));
                indexVarUbs.push_back(vOperand.ub(i));
                indexVarSteps.push_back(vOperand.step(i));
            }

            LoopNestBuilder(indexVarPtrs, indexVarLbs, indexVarUbs, indexVarSteps)([&] {
                IndexedValue ivRes(result);
                IndexedValue ivOperand(operand);

                // On the LHS of the assignment, adjust the index for the concatenation axis.
                llvm::SmallVector<ValueHandle, 5> resIndexHandles;
                for (int i = 0; i < rank; i++)
                {
                    resIndexHandles.push_back(i == concatenationAxis
                                                  ? indexVars[i] + concatenationAxisPos
                                                  : indexVars[i]);
                }

                ivRes(resIndexHandles) = ivOperand(indexVars);
            });

            // Move up concatenation_axis_pos for the next operand.
            concatenationAxisPos = concatenationAxisPos + vOperand.ub(concatenationAxis);
        }

        rewriter.replaceOp(op, {result});

        return matchSuccess();
    }

    REWRITER(NGReturnOp)
    {
        rewriter.replaceOpWithNewOp<ReturnOp>(op);
        return matchSuccess();
    }

#undef REWRITER

    template <typename OP>
    void lower_binary_elementwise(Operation* op,
                                  ArrayRef<Value*> operands,
                                  PatternRewriter& rewriter,
                                  DialectLoweringPass& m_pass)
    {
        auto loc = cast<OP>(op).getLoc();
        auto result = m_pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result->getType().isa<MemRefType>());
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
        LoopNestBuilder(pivs, lbs, ubs, steps)(
            // single stmt body
            [&] {
                if (isa<NGAddOp>(op))
                {
                    iRes(ivs) = iLHS(ivs) + iRHS(ivs);
                }
                else if (isa<NGSubOp>(op))
                {
                    iRes(ivs) = iLHS(ivs) - iRHS(ivs);
                }
                else if (isa<NGMulOp>(op))
                {
                    iRes(ivs) = iLHS(ivs) * iRHS(ivs);
                }
                else if (isa<NGDivOp>(op))
                {
                    iRes(ivs) = iLHS(ivs) / iRHS(ivs);
                }
                else if (isa<NGGreaterOp>(op))
                {
                    iRes(ivs) = ValueHandle(iLHS(ivs)) > ValueHandle(iRHS(ivs));
                }
                else if (isa<NGLessOp>(op))
                {
                    iRes(ivs) = ValueHandle(iLHS(ivs)) < ValueHandle(iRHS(ivs));
                }
                else if (isa<NGMaxOp>(op))
                {
                    iRes(ivs) =
                        edsc::intrinsics::select(ValueHandle(iLHS(ivs)) > ValueHandle(iRHS(ivs)),
                                                 ValueHandle(iLHS(ivs)),
                                                 ValueHandle(iRHS(ivs)));
                }
                else if (isa<NGMinOp>(op))
                {
                    iRes(ivs) =
                        edsc::intrinsics::select(ValueHandle(iLHS(ivs)) < ValueHandle(iRHS(ivs)),
                                                 ValueHandle(iLHS(ivs)),
                                                 ValueHandle(iRHS(ivs)));
                }
                else
                {
                    NGRAPH_CHECK(false, "Unsupported op");
                }
            });
        rewriter.replaceOp(op, {result});
    }

    template <typename RedOp>
    void lowerIndexReduction(Operation* op,
                             ArrayRef<Value*> operands,
                             PatternRewriter& rewriter,
                             DialectLoweringPass& m_pass)
    {
        static_assert(std::is_same<RedOp, NGArgMinRedOp>() || std::is_same<RedOp, NGArgMaxRedOp>(),
                      "Template parameter is not supported by lowerIndexReduction");

        RedOp redOp = cast<RedOp>(op);
        auto loc = redOp.getLoc();
        auto axesAttr = redOp.axes();

        NGRAPH_CHECK(axesAttr.size() == 1, "Index Reduction op should have one reduction axis");
        Attribute axisAttr = *axesAttr.begin();
        unsigned axis = axisAttr.dyn_cast<IntegerAttr>().getInt();

        NGRAPH_CHECK(operands.size() == 1 && operands[0] != nullptr,
                     "Expected one non-null operand in Index Reduction op");

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value* arg = operands[0];

        Value* result = m_pass.buildOutputDefs(op, rewriter)[0];

        // Views
        MemRefView vRes(result), vArg(arg);
        // Index Values
        IndexedValue iRes(result), iArg(arg);
        // Bounds Index Handles
        auto resLbs = vRes.getLbs();
        auto resUbs = vRes.getUbs();
        auto argLbs = vArg.getLbs();
        auto argUbs = vArg.getUbs();

        Type resTy = result->getType().cast<MemRefType>().getElementType();
        // Generate loop nest that initializes result to lower bound of the axis to be reduced.
        {
            auto ivs = IndexHandle::makeIndexHandles(vRes.rank());
            auto pivs = IndexHandle::makeIndexHandlePointers(ivs);
            auto steps = vRes.getSteps();
            auto initVal = vArg.lb(axis);
            LoopNestBuilder(pivs, resLbs, resUbs, steps)(
                [&] { iRes(ivs) = ValueHandle::create<IndexCastOp>(initVal, resTy); });
        }

        // Generate loop nest that computes the actual index reduction.
        {
            auto allIVs = IndexHandle::makeIndexHandles(vArg.rank());
            auto pAllIVs = IndexHandle::makeIndexHandlePointers(allIVs);
            auto steps = vArg.getSteps();
            SmallVector<IndexHandle, 8> nonRedIVs;

            Type resTy = result->getType().cast<MemRefType>().getElementType();
            NGRAPH_CHECK(resTy.isa<IntegerType>(),
                         "Expected integer result type in index reduction");

            // iterate over all argument dimensions
            LoopNestBuilder(pAllIVs, argLbs, argUbs, steps)([&] {
                // build a list of non-reduction IVs
                for (auto i = 0; i < vArg.rank(); i++)
                {
                    if (i != axis)
                        nonRedIVs.push_back(allIVs[i]);
                }

                // Load current min index with integer data type and convert it to index data type.
                ValueHandle currRedIdx = ValueHandle::create<IndexCastOp>(
                    (ValueHandle)iRes(nonRedIVs), IndexType::get(resTy.getContext()));

                // Build list of IVs including current min index.
                auto tempIVs = allIVs;
                tempIVs[axis] = currRedIdx;

                // Select the min/max value and cast it back to integer type before storing it.
                ValueHandle newRedIdx =
                    std::is_same<RedOp, NGArgMinRedOp>()
                        ? edsc::intrinsics::select(
                              iArg(allIVs) < iArg(tempIVs), allIVs[axis], currRedIdx)
                        : edsc::intrinsics::select(
                              iArg(tempIVs) < iArg(allIVs), allIVs[axis], currRedIdx);

                iRes(nonRedIVs) = ValueHandle::create<IndexCastOp>(newRedIdx, resTy);
            });
        }

        rewriter.replaceOp(op, result);
    }
}

namespace mlir
{
    Pass* createDialectLoweringPass(ngraph::runtime::ngmlir::MLIRCompiler* compiler)
    {
        return new DialectLoweringPass(*compiler);
    }
}
