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

#include "lowerer.hpp"

#include "dialect/ops.hpp"
#include "dialect/type.hpp"
#include "ngraph/assertion.hpp"

#include <llvm/ADT/DenseSet.h>
#include <mlir/EDSC/Builders.h>
#include <mlir/EDSC/Helpers.h>
#include <mlir/EDSC/Intrinsics.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <map>

#define PASS_NAME "convert-ngraph-to-affine"
#define DEBUG_TYPE PASS_NAME

// anonymous namespace
// no need to expose any of the following outside of this file
namespace
{
    using namespace mlir;
    using namespace mlir::edsc;
    using namespace mlir::edsc::op;
    using namespace ngraph::runtime;
    using namespace ngraph::runtime::ngmlir;
    // Index notation to generate standard (i.e., non-affine) loads and stores.
    using StdIndexedValue = TemplatedIndexedValue<intrinsics::std_load, intrinsics::std_store>;

    class DialectLoweringPass;

    /// Base class for nGraph operation conversions to affine/standard dialect. Provides
    /// conversion patterns with an access to the DialectLoweringPass which holds the state of the
    /// conversion.
    class NGraphOpLowering : public ConversionPattern
    {
    public:
        NGraphOpLowering(StringRef rootOpName, MLIRContext* context, DialectLoweringPass& pass)
            : ConversionPattern(rootOpName, /*benefit=*/1, context)
            , pass(pass){};

    protected:
        // Back-reference to the lowering pass which contains the lowering state, including the
        // nGraph type converter.
        DialectLoweringPass& pass;
    };

// Conversion classes declarations
#define MLIR_OP(OP, INPLACE)                                                                       \
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
                                           ConversionPatternRewriter& rewriter) const override;    \
    };

#include "op_lowerers.inc"

    // FuncOp Conversion pattern
    class FuncOpSignatureConversion : public ConversionPattern
    {
    public:
        FuncOpSignatureConversion(MLIRContext* ctx, TypeConverter& converter)
            : ConversionPattern(FuncOp::getOperationName(), 1, ctx)
            , converter(converter)
        {
        }

        /// Hook for derived classes to implement combined matching and rewriting.
        PatternMatchResult matchAndRewrite(Operation* op,
                                           ArrayRef<Value*> operands,
                                           ConversionPatternRewriter& rewriter) const override
        {
            auto funcOp = cast<FuncOp>(op);
            FunctionType type = funcOp.getType();

            // Convert the original function arguments.
            TypeConverter::SignatureConversion result(type.getNumInputs());
            for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
            {
                if (failed(converter.convertSignatureArg(i, type.getInput(i), result)))
                {
                    return matchFailure();
                }
            }

            auto funcTypeResults = type.getResults();
            if (!funcTypeResults.empty())
            {
                // Convert the original function results.
                SmallVector<Type, 4> convertedResults;
                if (failed(converter.convertTypes(funcTypeResults, convertedResults)))
                {
                    return matchFailure();
                }

                // Add result types as input args without mapping
                result.addInputs(convertedResults);
            }

            // Create a new function with an updated signature.
            auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
            rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(), newFuncOp.end());
            newFuncOp.setType(
                FunctionType::get(result.getConvertedTypes(), {/*void*/}, funcOp.getContext()));

            // Tell the rewriter to convert the region signature.
            rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
            rewriter.replaceOp(op, llvm::None);
            return matchSuccess();
        }

        /// The type converter to use when rewriting the signature.
        TypeConverter& converter;
    };

    // Helpers
    template <typename RedOp>
    void lowerIndexReduction(Operation* op,
                             ArrayRef<Value*> operands,
                             PatternRewriter& rewriter,
                             DialectLoweringPass& pass);

    template <typename OP>
    void lowerBinaryElementwise(Operation* op,
                                ArrayRef<Value*> operands,
                                PatternRewriter& rewriter,
                                DialectLoweringPass& pass);

    template <typename OP>
    void lowerUnaryElementwise(Operation* op,
                               ArrayRef<Value*> operands,
                               PatternRewriter& rewriter,
                               DialectLoweringPass& pass);

    ValueHandle createZeroConstant(mlir::Type type);

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
        void runOnModule() override;

        SmallVector<Value*, 4> buildOutputDefs(Operation* op, PatternRewriter& rewriter);
        Value* createTempTensor(Type type, PatternRewriter& rewriter);

        /// Inserts dealloc Ops for each temporary allocated by AllocOp
        void insertDeallocs(PatternRewriter& rewriter);

        NGraphTypeConverter& getTypeConverter() { return typeConverter; }
    private:
        /// Collect a set of patterns to convert from the nGraph dialect to Affine dialect.
        void populateNGraphToAffineConversionPatterns(OwningRewritePatternList& patterns);

        void findOutputValues();
        void insertNoAliasArgAttrs();

    private:
        NGraphTypeConverter typeConverter;
        // List of temporary memrefs to deallocate at end of function
        SmallVector<Value*, 4> memRefsToDealloc;

        // Ops maybe assigned mem-refs in previous memory optimization passes.
        // Track pre-assigned buffers  for each Value and re-use it if one is available.
        using IdToMemRefMap = std::unordered_map<unsigned, Value*>;
        IdToMemRefMap m_id_to_memref;

        // TODO: Workaround for findOutputValues and buildOutputDefs. See NGCPU-470.
        std::string funcName;
    };

    void DialectLoweringPass::runOnModule()
    {
        // Create type converter and initialize conversion patterns.
        NGraphTypeConverter converter;
        OwningRewritePatternList patterns;

        populateNGraphToAffineConversionPatterns(patterns);

        // Create target that defines legal ops for nGraph dialect to be lowered to.
        ConversionTarget target(getContext());

        target.addLegalDialect<AffineOpsDialect, StandardOpsDialect>();
        target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
        target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
            // FuncOp is legal only if types have been converted to Std types.
            return typeConverter.isSignatureLegal(op.getType());
        });

        // Gather functions to be processed. Note that new functions will be added to module as part
        // of the function signature conversion so we have to collect the original ones before hand.
        SmallVector<FuncOp, 2> origFuncOps(getModule().getOps<FuncOp>());

        for (auto origFunc : origFuncOps)
        {
            // TODO: Workaround for findOutputValues and buildOutputDefs. See NGCPU-470.
            funcName = origFunc.getName();

            // Capture output values by looking for the Return and grabbing the values the order of
            // the returned values matches the order of the lowered func signature for results. This
            // is used to find the arg_id that a defined value maps to if it is an output.
            findOutputValues();

            // NOTE: Function signature conversion creates a new FuncOp that is inserted in the
            // module. References the original FuncOp are no longer valid after this point.
            if (failed(applyFullConversion(origFunc, target, std::move(patterns), &converter)))
            {
                emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering nGraph dialect\n");
                signalPassFailure();
            }

            // TODO: Encode no alias attribute as part of the function signature conversion or as a
            // separate rewrite pattern. Retrieve new function after signature conversion.
            insertNoAliasArgAttrs();
        }
    }

    void DialectLoweringPass::populateNGraphToAffineConversionPatterns(
        OwningRewritePatternList& patterns)
    {
#define MLIR_OP(OP, INPLACE) OP##Conversion,
#define MLIR_LAST_OP(OP, INPLACE) OP##Conversion
        patterns.insert<
#include "op_lowerers.inc"
            >(&getContext(), *this);

        // FuncOp pattern
        patterns.insert<FuncOpSignatureConversion>(&getContext(), typeConverter);
    }

    void DialectLoweringPass::findOutputValues()
    {
        FuncOp f = getModule().lookupSymbol<mlir::FuncOp>(funcName);
        NGRAPH_CHECK(f, "FuncOp '" + funcName + "' not found");

        SmallVector<Value*, 4> outputList;
        unsigned outputCount = 0;
        unsigned inputCount = f.getType().getNumInputs();
        // we find out output values by looking at returned values
        // any return should return all outputs of the subgraph
        f.walk([this, &outputCount, inputCount](NGReturnOp ret) {
            for (unsigned i = 0; i < ret.getNumOperands(); i++)
            {
                // annotate instructions defining outputs with the arg idx of the output
                auto outputValue = ret.getOperand(i);
                auto op = outputValue->getDefiningOp();

                op->setAttr(
                    "graphOutputIdx",
                    mlir::IntegerAttr::get(IntegerType::get(32, op->getContext()), i + inputCount));
            }
            NGRAPH_CHECK(outputCount == 0 || outputCount == ret.getNumOperands(),
                         "Inconsistent returns in function");
        });
    }

    SmallVector<Value*, 4> DialectLoweringPass::buildOutputDefs(Operation* op,
                                                                PatternRewriter& rewriter)
    {
        FuncOp f = getModule().lookupSymbol<mlir::FuncOp>(funcName);
        NGRAPH_CHECK(f, "FuncOp '" + funcName + "' not found");

        SmallVector<Value*, 4> newResults;
        for (auto origResult : op->getResults())
        {
            // find output arg if this operation produces any sub-graph outputs
            if (IntegerAttr attr = op->getAttrOfType<IntegerAttr>("graphOutputIdx"))
            {
                mlir::Block* entryBlock = &*(f.begin());
                unsigned argId = (unsigned)attr.getInt();
                newResults.push_back(entryBlock->getArgument(argId));
            }
            else
            {
                auto tensorType = origResult->getType().cast<NGTensorType>();
                Value* newResult;
                Attribute bufferIdAttr = getBufferId(op);
                if (!bufferIdAttr)
                {
                    // Allocate new memref
                    newResult = createTempTensor(typeConverter.convertType(tensorType), rewriter);
                }
                else
                {
                    unsigned bufferId = bufferIdAttr.cast<IntegerAttr>().getInt();
                    // Re-use a memref if it exist, else create a new one and update map
                    IdToMemRefMap::iterator it = m_id_to_memref.find(bufferId);
                    if (it == m_id_to_memref.end())
                    {
                        // create a new memref
                        newResult =
                            createTempTensor(typeConverter.convertType(tensorType), rewriter);
                        m_id_to_memref[bufferId] = newResult;
                    }
                    else
                    {
                        newResult = it->second;
                    }
                }
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
        memRefsToDealloc.push_back(alloc);

        // TODO:
        // Enable dynamic memref allocation via call-back to nGraph allocator
        // We should create a list of Values representing each dynamic dim
        // The values would be computed based on the shape of the input to the ng op we are
        // lowering.
        // E.g. If lowering concat, Value for dynamic concat axis will be the sum of input dims.
        // The lowerer will generate code to compute the dims.
        // This is better be done via std.AllocOp but we need to make it hookable to nGraph
        // allocator call-back.

        return alloc;
    }

    /// Add llvm.noalias attribute to all the memref function arguments. We know that this is safe
    /// by nGraph op semantics.
    void DialectLoweringPass::insertNoAliasArgAttrs()
    {
        FuncOp func = getModule().lookupSymbol<mlir::FuncOp>(funcName);
        NGRAPH_CHECK(func, "FuncOp '" + funcName + "' not found");

        unsigned int argIdx = 0;
        for (auto* arg : func.getArguments())
        {
            if (arg->getType().isa<MemRefType>())
            {
                func.setArgAttr(argIdx, "llvm.noalias", BoolAttr::get(true, &getContext()));
            }

            ++argIdx;
        }
    }

    void DialectLoweringPass::insertDeallocs(PatternRewriter& rewriter)
    {
        for (auto value : memRefsToDealloc)
        {
            rewriter.create<DeallocOp>(rewriter.getUnknownLoc(), value);
        }
    }

    // NGDialect converters
    Type NGraphTypeConverter::convertType(Type type)
    {
        // We may need to refactor this code to a external utility if type conversion is needed
        // outside of the lowering context since NGraphTypeConverter is private.

        if (auto tensorType = type.dyn_cast<NGTensorType>())
        {
            // Convert NGTensorType to Std MemRefType directly instead of going to Std TensorType.
            // This may change in the future.
            return MemRefType::get(tensorType.getShape(),
                                   convertType(tensorType.getElementType()),
                                   {/* no map used */},
                                   0);
        }
        if (auto floatType = type.dyn_cast<NGFloatType>())
        {
            // Float types are already std type.
            return floatType;
        }
        if (auto intType = type.dyn_cast<NGIntegerType>())
        {
            return mlir::IntegerType::get(intType.getWidth(), intType.getContext());
        }
        if (auto boolType = type.dyn_cast<NGBoolType>())
        {
            return mlir::IntegerType::get(1 /* width */, boolType.getContext());
        }

        // Do not assert/NGRAPH_CHECK here. Type convertion infra expects `convertType` to return
        // the input type if the type is not supported.
        return type;
    }

#define REWRITER(OP)                                                                               \
    PatternMatchResult OP##Conversion::matchAndRewrite(                                            \
        Operation* op, ArrayRef<Value*> operands, ConversionPatternRewriter& rewriter) const

    REWRITER(NGAddOp)
    {
        lowerBinaryElementwise<mlir::NGAddOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGSubOp)
    {
        lowerBinaryElementwise<mlir::NGSubOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGMulOp)
    {
        lowerBinaryElementwise<mlir::NGMulOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGDivOp)
    {
        lowerBinaryElementwise<mlir::NGDivOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGGreaterOp)
    {
        lowerBinaryElementwise<mlir::NGGreaterOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGLessOp)
    {
        lowerBinaryElementwise<mlir::NGLessOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGMaxOp)
    {
        lowerBinaryElementwise<mlir::NGMaxOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGMinOp)
    {
        lowerBinaryElementwise<mlir::NGMinOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGArgMaxRedOp)
    {
        lowerIndexReduction<mlir::NGArgMaxRedOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGArgMinRedOp)
    {
        lowerIndexReduction<mlir::NGArgMinRedOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    // Relu
    REWRITER(NGReluOp)
    {
        auto loc = cast<NGReluOp>(op).getLoc();

        auto result = pass.buildOutputDefs(op, rewriter)[0];
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
        auto ivs = makeIndexHandles(vLHS.rank());
        auto pivs = makeIndexHandlePointers(ivs);
        // Steps
        auto steps = vLHS.getSteps();

        NGRAPH_CHECK(lhs->getType().isa<MemRefType>());
        Type elemTy = lhs->getType().dyn_cast<MemRefType>().getElementType();

        LoopNestBuilder(pivs, lbs, ubs, steps)([&] {
            ValueHandle val = iLHS(ivs);
            ValueHandle zero = createZeroConstant(elemTy);
            iRes(ivs) = intrinsics::select(val > zero, val, zero);
        });

        rewriter.replaceOp(op, {result});
        return matchSuccess();
    }

    // Negative
    REWRITER(NGNegOp)
    {
        lowerUnaryElementwise<mlir::NGNegOp>(op, operands, rewriter, pass);
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
        Value* result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs && rhs && result, "Unexpected null values in DotOp");

        auto resultTy = result->getType().dyn_cast<MemRefType>();
        auto lhsTy = lhs->getType().dyn_cast<MemRefType>();
        auto rhsTy = rhs->getType().dyn_cast<MemRefType>();
        NGRAPH_CHECK(resultTy, "Unexpected non-memref result type");
        NGRAPH_CHECK(lhsTy, "Unexpected non-memref LHS type");
        NGRAPH_CHECK(rhsTy, "Unexpected non-memref RHS type");

        Type elemTy = resultTy.getElementType();
        NGRAPH_CHECK(elemTy == lhsTy.getElementType() && elemTy == rhsTy.getElementType(),
                     "Types mismatch in DotOp");

        // Create the following loop nest for matmul operation:
        //   for(n, N, 1)
        //     for(m, M, 1)
        //       for(k, K, 1)
        //         res[n, k] += lhs[n, m] * rhs[m, k]
        // TODO (dcab): We currently generate a super naive loop nest. Improve loop nest layout.

        MemRefView vRes(result), vLhs(lhs), vRhs(rhs);

        NGRAPH_CHECK(vLhs.rank() == 2 && vRhs.rank() == 2 && vRes.rank() == 2,
                     "Dot operation is only supported for 2D tensors");

        // Create induction variables, lower bounds, upper bounds and steps of the loop nest.
        // It's important to note that MemRefView priovides lb/ub/step info is "reverse order",
        // i.e., fastest varying dimension is the last one, slowest varying dimention is the first
        // one.
        IndexHandle n, m, k;
        unsigned nDim = vLhs.fastestVarying() - 1;
        unsigned mDim = vRhs.fastestVarying();
        unsigned kDim = vRhs.fastestVarying();
        IndexHandle nLb(vLhs.lb(nDim)), mLb(vLhs.lb(mDim)), kLb(vRhs.lb(kDim));
        IndexHandle nUb(vLhs.ub(nDim)), mUb(vLhs.ub(mDim)), kUb(vRhs.ub(kDim));
        int64_t nStep = vLhs.step(nDim), mStep = vLhs.step(mDim), kStep = vRhs.step(kDim);

        // Constants and indexed values to be used inside the loop nest.
        IndexedValue iRes(result), iLhs(lhs), iRhs(rhs);
        ValueHandle zeroInit(rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elemTy)));

        {
            IndexHandle n, k;
            LoopBuilder(&n, nLb, nUb, nStep)(
                [&] { LoopBuilder(&k, kLb, kUb, kStep)([&] { iRes(n, k) = zeroInit; }); });
        }
        LoopBuilder(&n, nLb, nUb, nStep)([&] {
            LoopBuilder(&m, mLb, mUb, mStep)([&] {
                LoopBuilder(&k, kLb, kUb, kStep)([&] { iRes(n, k) += iLhs(n, m) * iRhs(m, k); });
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
        Value* result = pass.buildOutputDefs(op, rewriter)[0];
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

    REWRITER(NGGatherOp)
    {
        auto gatherOp = cast<NGGatherOp>(op);
        auto loc = gatherOp.getLoc();
        ScopedContext scope(rewriter, loc);

        // Get operands
        Value* result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in GatherOp");

        Value* params = operands[0];
        Value* indices = operands[1];
        auto axis = gatherOp.axis().getSExtValue();

        // Create view to write into result.
        MemRefView vRes(result), vParams(params), vIndices(indices);
        // Indexed Values
        IndexedValue iRes(result), iIndices(indices);
        StdIndexedValue iParams(params);

        // Construct outer loop for params dims. Exclude the axis dim.
        SmallVector<ValueHandle, 4> paramsLbs, paramsUbs;
        SmallVector<IndexHandle, 4> paramsIVs;
        SmallVector<int64_t, 4> paramsSteps;
        SmallVector<ValueHandle*, 4> paramsIVPtrs;
        for (auto i = 0; i < vParams.rank(); i++)
        {
            // skip gather axis
            if (i == axis)
                continue;
            paramsLbs.push_back(IndexHandle(vParams.lb(i)));
            paramsUbs.push_back(IndexHandle(vParams.ub(i)));
            paramsSteps.push_back(vParams.step(i));
        }
        NGRAPH_CHECK(paramsLbs.size() == vParams.rank() - 1 &&
                         paramsUbs.size() == paramsLbs.size() &&
                         paramsSteps.size() == paramsLbs.size(),
                     "Incorrect loop nest bounds size for gather params");

        paramsIVs = makeIndexHandles(vParams.rank() - 1);
        paramsIVPtrs = makeIndexHandlePointers(paramsIVs);

        auto indicesLbs = vIndices.getLbs();
        auto indicesUbs = vIndices.getUbs();
        auto indicesSteps = vIndices.getSteps();

        auto indicesIVs = makeIndexHandles(vIndices.rank());
        auto indicesIVPtrs = makeIndexHandlePointers(indicesIVs);

        SmallVector<IndexHandle, 8> paramsIndices, resIndices;

        // Make sure we are going to create loops
        NGRAPH_CHECK(vParams.rank() > 0, "Invalid size for indices steps");

        // Let params rank : N
        // Let indices rank : M
        // Let axis be A
        // Generate
        // indices loops
        // for I_0:0 -> indices.dim[0]
        // ...
        //   for I_(M-1):0 -> indices.dim[M-1]
        //     params loops
        //     for P_0: 0 -> params.dim[0]
        //       for P_1: 0 -> params.dim[1]
        //         for P_2: 0 -> params.dim[2]
        // ...
        //           for P_(A-1):0 -> params.dim[A-1]
        //             for P_(A+1):0 -> params.dim[A+1]
        // ...
        //               for P_(N-1):0 -> params.dim[N-1]
        //                 res[P_0, P_1, .. P_(A-1), I_0, .., I_(M-1), P_(A+1), ... P_(N-1)] =
        //                   params[P_0, P_1, .. P_(A-1), indices[I_0, .., I_(M-1)],
        //                          P_(A+1), ... P_(N-1)];

        LoopNestBuilder(indicesIVPtrs, indicesLbs, indicesUbs, indicesSteps)([&] {
            // Load axis value from indices array and cast it to Index Type
            ValueHandle axisIdx = ValueHandle::create<IndexCastOp>(
                (ValueHandle)iIndices(indicesIVs), rewriter.getIndexType());

            LoopNestBuilder(paramsIVPtrs, paramsLbs, paramsUbs, paramsSteps)([&] {
                // construct indices for param
                // [P_0, P_1, .. P_axis-1, Indices[I0, I1, .. I_k-1], P_axis+1, P_axis+2, .. P_n-1]
                for (auto i = 0, j = 0; i < vParams.rank(); i++)
                {
                    if (i == axis)
                    {
                        paramsIndices.push_back(IndexHandle(axisIdx));
                    }
                    else
                    {
                        paramsIndices.push_back(paramsIVs[j++]);
                    }
                }

                // construct indices for result
                // [P_0, P_1, .. P_axis-1, I0, I1, .. I_k-1, P_axis+1, P_axis+2, .. P_n-1]
                for (auto i = 0, j = 0; i < vParams.rank() + vIndices.rank() - 1;)
                {
                    if (i == axis && indicesIVs.size() > 0)
                    {
                        resIndices.append(indicesIVs.begin(), indicesIVs.end());
                        i += indicesIVs.size();
                    }
                    else
                    {
                        resIndices.push_back(paramsIVs[j++]);
                        i++;
                    }
                }
                // Store into result
                iRes(resIndices) = iParams(paramsIndices);
            });
        });

        rewriter.replaceOp(op, {result});
        return matchSuccess();
    }

    REWRITER(NGConvolutionOp)
    {
        auto convolOp = cast<NGConvolutionOp>(op);
        auto loc = convolOp.getLoc();
        ScopedContext scope(rewriter, loc);

        // Get operands
        Value* result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in Convolution Op");
        Value* images = operands[0];
        Value* filters = operands[1];
        auto strides = convolOp.strides().getValue();
        auto padBelow = convolOp.padBelow().getValue();
        auto padAbove = convolOp.padBelow().getValue();

        Type elemTy = images->getType().cast<MemRefType>().getElementType();

        // Let Images shape be  [N, C_IN, D_1, ... D_f]
        // Let Filters shape be [C_OUT, C_IN, F_1, ... F_f]
        // Output shape will be [N, C_OUT, R_1, ..R_f]
        //   where R_i = (AdjD_i - AdjF_i + 1) / Strides[i]
        //
        // AdjD_i is adjusted image spatial dimension after padding and dilation
        //   AdjD_i = padBelow[i] + (dilation[i] * (D_i - 1) + 1) + padAbove[i]
        //
        // AdjF_i is adjusted filters spatial dimension after dilation
        //   AdjF_i = dilation[i] * (F_i - 1) + 1
        //
        //   If no padding, padAbove/Below[i] = 0
        //   If no dilation, dilation[i] is 1
        //
        // Generate the following (currently without padding/dilation support)
        //
        //
        // for n : 0 -> N
        //   for k : 0 -> C_OUT
        //     for <r_1 .. r_f> : <0 .. 0> -> <R_1 ... R_f>
        //       //initialize result to zero
        //       Output[n, k, r_1, .. r_f] = 0;
        //
        // for n : 0 -> N
        //   for k : 0 -> C_OUT
        //     for c : 0 -> C_IN
        //       // iterate over output spatial shape
        //       for <r_1 .. r_f> : <0 .. 0> -> <R_1 ... R_f> //
        //         //compute image start inputs indices
        //         i_1 = r_1 * strides[0];
        //         ..
        //         i_f = r_f * strides[f - 1];
        //         // iterate over kernel spatial shape
        //         for <j_1 .. j_f> : <0 .. 0> -> <F_1 .. F_f>
        //           Output[n, k, r_1, .. r_f] +=
        //             Images[n, c, i_1 + j_1, .. i_f + j_f] * Filters[k, c, j_1, .. j_f]

        // With padding, we check (using IntegerSets) whether each spatial dim in Images lie inside
        // non-padded spatial region. If true, we perform the computation:
        //
        //         for <j_1 .. j_f> : <0 .. 0> -> <F_1 .. F_f>
        //         if(indices in non-padded region):
        //           Output[n, k, r_1, .. r_f] +=
        //             Images[n, c, i_1 + j_1, .. i_f + j_f] * Filters[k, c, j_1, .. j_f]

        // Create view to write into result.
        MemRefView vRes(result), vImages(images), vFilters(filters);

        // Indexed Values
        IndexedValue iRes(result), iImages(images), iFilters(filters);

        // Bounds on batch size N
        ValueHandle batchLb = vImages.lb(0), batchUb = vImages.ub(0);
        // Bounds on number of filters
        ValueHandle numFiltersLb = vFilters.lb(0), numFiltersUb = vFilters.ub(0);
        // Bound on number of channels
        ValueHandle numChannelsLb = vImages.lb(1), numChannelsUb = vImages.ub(1);
        // Bounds on result spatial dimensions
        SmallVector<ValueHandle, 4> resSpatialLbs, resSpatialUbs;
        SmallVector<ValueHandle, 4> imgSpatialLbs, imgSpatialUbs;
        SmallVector<ValueHandle, 4> filtersSpatialLbs, filtersSpatialUbs;
        // Spatial rank
        unsigned spatialRank = vImages.rank() - 2;

        // Result spatial indices and bounds
        auto resSpatialIndices = makeIndexHandles(spatialRank);
        auto resSpatialIndicesPtrs = makeIndexHandlePointers(resSpatialIndices);
        SmallVector<int64_t, 4> resSteps, filtersSteps;
        SmallVector<int, 4> padBelowIntValues;
        bool withPadding = false;

        for (auto i = 0; i < spatialRank; i++)
        {
            // result spatial bounds and steps
            resSpatialLbs.push_back(vRes.lb(i + 2));
            resSpatialUbs.push_back(vRes.ub(i + 2));
            resSteps.push_back(vRes.step(i + 2));
            // image spatial bounds
            imgSpatialLbs.push_back(vImages.lb(i + 2));
            imgSpatialUbs.push_back(vImages.ub(i + 2));

            // Check if we have any padding and collect pad values
            IntegerAttr iAttr = padBelow[i].cast<IntegerAttr>();
            int padValue = iAttr.getInt();
            if (padValue)
            {
                withPadding = true;
            }
            padBelowIntValues.push_back(padValue);

            iAttr = padAbove[i].cast<IntegerAttr>();
            padValue = iAttr.getInt();
            if (padValue)
            {
                withPadding = true;
            }
        }

        NGRAPH_CHECK(vImages.rank() == vFilters.rank(), "Images and Filters have unequal ranks");
        NGRAPH_CHECK(resSpatialLbs.size() == resSpatialUbs.size() &&
                         resSpatialLbs.size() == spatialRank,
                     "Results spatial dims mismatches input");

        // Filters spatial indices and bounds
        auto filtersSpatialIndices = makeIndexHandles(spatialRank);
        auto filtersSpatialIndicesPtrs = makeIndexHandlePointers(filtersSpatialIndices);

        for (auto i = 0; i < spatialRank; i++)
        {
            filtersSpatialLbs.push_back(vFilters.lb(i + 2));
            filtersSpatialUbs.push_back(vFilters.ub(i + 2));
            filtersSteps.push_back(vFilters.step(i + 2));
        }

        IntegerSet nonPaddedRange;
        if (withPadding)
        {
            // Create affine expressions and IntegerSet
            // IntegerSet (d0, d1, .. d_N-1)[LB_0, LB_1, .. LB_N-1, UB_0, UB_1, .. UB_N-1], where
            // for each dim:
            //   (d_dim - padBelow[dim] - LB_dim >= 0),
            //   (padBelow[dim] + UB_dim - d_dim - 1 >= 0)
            SmallVector<AffineExpr, 4> affineExprs;
            // Bool to indicate if expr is equality or inequality
            SmallVector<bool, 4> isEq;

            for (unsigned dim = 0; dim < spatialRank; dim++)
            {
                // i_dim
                auto dimExpr = rewriter.getAffineDimExpr(dim);
                auto imgLbExpr = rewriter.getAffineSymbolExpr(dim);

                // expr1 : i_dim - padBelow[dim] - imgLB >= 0
                auto padBelowExpr = rewriter.getAffineConstantExpr(padBelowIntValues[dim]);
                affineExprs.push_back(dimExpr - padBelowExpr - imgLbExpr);
                isEq.push_back(false);

                // expr2: padBelow[dim] + imgUB - i_dim - 1 >= 0
                auto imgUbExpr = rewriter.getAffineSymbolExpr(spatialRank + dim);
                auto oneExpr = rewriter.getAffineConstantExpr(1);
                affineExprs.push_back(padBelowExpr + imgUbExpr - dimExpr - oneExpr);
                isEq.push_back(false);
            }

            NGRAPH_CHECK(affineExprs.size() == isEq.size() && isEq.size() == 2 * spatialRank,
                         "Invalid number of expressions in the IntegerSet");
            nonPaddedRange =
                rewriter.getIntegerSet(spatialRank, 2 * spatialRank, affineExprs, isEq);
        }

        // Initialize output to zero
        {
            IndexHandle n, k, c;
            auto resSpatialIndices = makeIndexHandles(spatialRank);
            auto resSpatialIndicesPtrs = makeIndexHandlePointers(resSpatialIndices);

            LoopBuilder(&n, batchLb, batchUb, 1)([&] {
                LoopBuilder(&k, numFiltersLb, numFiltersUb, 1)([&] {
                    LoopNestBuilder(
                        resSpatialIndicesPtrs, resSpatialLbs, resSpatialUbs, resSteps)([&] {
                        SmallVector<IndexHandle, 4> resIndices;
                        // Result indices
                        resIndices.push_back(n);
                        resIndices.push_back(k);
                        resIndices.insert(
                            resIndices.end(), resSpatialIndices.begin(), resSpatialIndices.end());
                        ValueHandle zero = createZeroConstant(elemTy);
                        iRes(resIndices) = zero;
                    });
                });
            });
        }

        IndexHandle n, k, c;
        // Convolution loop
        LoopBuilder(&n, batchLb, batchUb, 1)([&] {
            // Number of filters loop
            LoopBuilder(&k, numFiltersLb, numFiltersUb, 1)([&] {
                // Channels loop
                LoopBuilder(&c, numChannelsLb, numChannelsUb, 1)([&] {
                    // Results loop
                    LoopNestBuilder(
                        resSpatialIndicesPtrs, resSpatialLbs, resSpatialUbs, resSteps)([&] {
                        // Compute image start indices
                        SmallVector<IndexHandle, 4> imgStartIndices;
                        for (auto i = 0; i < spatialRank; i++)
                        {
                            IntegerAttr iAttr = strides[i].cast<IntegerAttr>();
                            auto stride = intrinsics::constant_index(iAttr.getInt());
                            imgStartIndices.push_back(IndexHandle(resSpatialIndices[i] * stride));
                        }
                        SmallVector<IndexHandle, 4> resIndices;
                        // Result indices
                        resIndices.push_back(n);
                        resIndices.push_back(k);
                        resIndices.insert(
                            resIndices.end(), resSpatialIndices.begin(), resSpatialIndices.end());
                        // Filters spatial loop
                        LoopNestBuilder(filtersSpatialIndicesPtrs,
                                        filtersSpatialLbs,
                                        filtersSpatialUbs,
                                        filtersSteps)([&] {
                            SmallVector<IndexHandle, 4> imgIndices, filtersIndices;
                            // Image indices
                            // Here we compute the virtual start index into the padded image.

                            imgIndices.push_back(n);
                            imgIndices.push_back(c);
                            for (auto i = 0; i < spatialRank; i++)
                            {
                                imgIndices.push_back(
                                    IndexHandle(imgStartIndices[i] + filtersSpatialIndices[i]));
                            }
                            // Filter indices
                            filtersIndices.push_back(k);
                            filtersIndices.push_back(c);
                            filtersIndices.insert(filtersIndices.end(),
                                                  filtersSpatialIndices.begin(),
                                                  filtersSpatialIndices.end());

                            if (withPadding)
                            {
                                // if args : img dims, img lbs, img ubs
                                SmallVector<IndexHandle, 4>::iterator it = imgIndices.begin();
                                std::advance(it, 2);
                                SmallVector<Value*, 4> affineIfArgs(it, imgIndices.end());
                                affineIfArgs.insert(
                                    affineIfArgs.end(), imgSpatialLbs.begin(), imgSpatialLbs.end());
                                affineIfArgs.insert(
                                    affineIfArgs.end(), imgSpatialUbs.begin(), imgSpatialUbs.end());

                                auto affineIfOp =
                                    rewriter.create<AffineIfOp>(rewriter.getUnknownLoc(),
                                                                nonPaddedRange,
                                                                affineIfArgs,
                                                                /*withElseRegion=*/false);
                                {
                                    auto rewriter = affineIfOp.getThenBodyBuilder();
                                    ScopedContext scope(rewriter, loc);
                                    // We must subtract pad below before img load, since the
                                    // physical image is not padded
                                    SmallVector<IndexHandle, 4> adjustedImgIndices;
                                    adjustedImgIndices.push_back(n);
                                    adjustedImgIndices.push_back(c);
                                    for (auto i = 0; i < spatialRank; i++)
                                    {
                                        adjustedImgIndices.push_back(IndexHandle(
                                            imgIndices[2 + i] -
                                            intrinsics::constant_index(padBelowIntValues[i])));
                                    }
                                    iRes(resIndices) =
                                        iRes(resIndices) +
                                        (iImages(adjustedImgIndices) * iFilters(filtersIndices));
                                }
                            }
                            else
                            {
                                iRes(resIndices) = iRes(resIndices) +
                                                   (iImages(imgIndices) * iFilters(filtersIndices));
                            }
                        });
                    });
                });
            });
        });

        rewriter.replaceOp(op, {result});
        return matchSuccess();
    }

    REWRITER(NGReturnOp)
    {
        pass.insertDeallocs(rewriter);
        rewriter.replaceOpWithNewOp<ReturnOp>(op);
        return matchSuccess();
    }

#undef REWRITER
    /// End of pattern matchers
    template <typename OP>
    void lowerUnaryElementwise(Operation* op,
                               ArrayRef<Value*> operands,
                               PatternRewriter& rewriter,
                               DialectLoweringPass& pass)
    {
        auto loc = cast<OP>(op).getLoc();

        auto result = pass.buildOutputDefs(op, rewriter)[0];
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
        auto ivs = makeIndexHandles(vLHS.rank());
        auto pivs = makeIndexHandlePointers(ivs);
        // Steps
        auto steps = vLHS.getSteps();

        NGRAPH_CHECK(lhs->getType().isa<MemRefType>());
        Type elemTy = lhs->getType().cast<MemRefType>().getElementType();

        LoopNestBuilder(pivs, lbs, ubs, steps)([&] {
            ValueHandle val = iLHS(ivs);
            if (isa<NGNegOp>(op))
            {
                ValueHandle zero = createZeroConstant(elemTy);
                iRes(ivs) = zero - val;
            }
            else
            {
                NGRAPH_CHECK(false, "Unsupported op");
            }
        });

        rewriter.replaceOp(op, {result});
    }

    template <typename OP>
    void lowerBinaryElementwise(Operation* op,
                                ArrayRef<Value*> operands,
                                PatternRewriter& rewriter,
                                DialectLoweringPass& pass)
    {
        auto loc = cast<OP>(op).getLoc();
        auto result = pass.buildOutputDefs(op, rewriter)[0];
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
        auto ivs = makeIndexHandles(vLHS.rank());
        auto pivs = makeIndexHandlePointers(ivs);
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
                             DialectLoweringPass& pass)
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

        Value* result = pass.buildOutputDefs(op, rewriter)[0];

        // Views
        MemRefView vRes(result), vArg(arg);
        // Index Values
        StdIndexedValue iRes(result), stdArg(arg);
        IndexedValue affineArg(arg);
        // Bounds Index Handles
        auto resLbs = vRes.getLbs();
        auto resUbs = vRes.getUbs();
        auto argLbs = vArg.getLbs();
        auto argUbs = vArg.getUbs();

        Type resTy = result->getType().cast<MemRefType>().getElementType();
        // Generate loop nest that initializes result to lower bound of the axis to be reduced.
        {
            auto ivs = makeIndexHandles(vRes.rank());
            auto pivs = makeIndexHandlePointers(ivs);
            auto steps = vRes.getSteps();
            auto initVal = vArg.lb(axis);
            LoopNestBuilder(pivs, resLbs, resUbs, steps)(
                [&] { iRes(ivs) = ValueHandle::create<IndexCastOp>(initVal, resTy); });
        }

        // Generate loop nest that computes the actual index reduction.
        {
            auto allIVs = makeIndexHandles(vArg.rank());
            auto pAllIVs = makeIndexHandlePointers(allIVs);
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
                    {
                        nonRedIVs.push_back(allIVs[i]);
                    }
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
                              affineArg(allIVs) < stdArg(tempIVs), allIVs[axis], currRedIdx)
                        : edsc::intrinsics::select(
                              stdArg(tempIVs) < affineArg(allIVs), allIVs[axis], currRedIdx);

                iRes(nonRedIVs) = ValueHandle::create<IndexCastOp>(newRedIdx, resTy);
            });
        }

        rewriter.replaceOp(op, result);
    }

    ValueHandle createZeroConstant(mlir::Type type)
    {
        if (auto floatTy = type.dyn_cast<FloatType>())
        {
            if (floatTy.isF32())
            {
                return intrinsics::constant_float(llvm::APFloat(0.0f), floatTy);
            }
            else if (floatTy.isF64())
            {
                return intrinsics::constant_float(llvm::APFloat(0.0), floatTy);
            }
            else
            {
                NGRAPH_UNREACHABLE("Unsupported floating-point precision");
            }
        }
        else if (auto intTy = type.dyn_cast<IntegerType>())
        {
            return intrinsics::constant_int(0, intTy.getWidth());
        }
        NGRAPH_UNREACHABLE("Unsupported type");
    }
}

namespace mlir
{
    std::unique_ptr<Pass> createDialectLoweringPass()
    {
        return std::make_unique<DialectLoweringPass>();
    }
} // namespace mlir

static PassRegistration<DialectLoweringPass> pass(PASS_NAME,
                                                  "Convert nGraph dialect to affine dialect");
