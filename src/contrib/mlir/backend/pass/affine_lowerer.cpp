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

#include "affine_lowerer.hpp"

#include "contrib/mlir/backend/analysis/memory_analysis.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/type.hpp"
#include "contrib/mlir/runtime/cpu/callback_utils.hpp"
#include "contrib/mlir/utils.hpp"
#include "ngraph/assertion.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/AffineOps/EDSC/Builders.h>
#include <mlir/Dialect/AffineOps/EDSC/Intrinsics.h>
#include <mlir/Dialect/StandardOps/EDSC/Intrinsics.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/Function.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/StandardTypes.h>
#include <mlir/Transforms/DialectConversion.h>

#include <map>

#define PASS_NAME "convert-ngraph-to-affine"
#define DEBUG_TYPE PASS_NAME

// Enable the lowering of MemRefs to LLVM bare pointers.
extern llvm::cl::opt<bool> clEnableBarePtrMemRefLowering;

std::vector<ngraph::runtime::ngmlir::opAttrs> opAttrsVec;

// anonymous namespace
// no need to expose any of the following outside of this file
namespace
{
    using namespace mlir;
    using namespace mlir::edsc;
    using namespace mlir::edsc::intrinsics;
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
                                           ArrayRef<Value> operands,                               \
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
                                           ArrayRef<Value> operands,
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
                             ArrayRef<Value> operands,
                             PatternRewriter& rewriter,
                             DialectLoweringPass& pass);

    template <typename OP>
    void lowerBinaryElementwise(Operation* op,
                                ArrayRef<Value> operands,
                                PatternRewriter& rewriter,
                                DialectLoweringPass& pass);

    template <typename OP>
    void lowerUnaryElementwise(Operation* op,
                               ArrayRef<Value> operands,
                               PatternRewriter& rewriter,
                               DialectLoweringPass& pass);

    // Generates a convolution kernel that can be used to generate single or
    // group convolution. It can handle filters where C_OUT dim includes
    // all groups, or if groups is an additional dimension before C_OUT.
    //
    // For single convolution, the default variables do not
    // have to be specific and will be auto-deduced from the input shapes.
    //
    // For group convolution, the caller has to generate the outer loop
    // over the number of groups. It will also generate the bounds on the
    // C_IN and C_OUT dimensions. It will pass the bounds and IV of the outer
    // loop as follows:
    //
    // cLb/Ub : Values representing bounds on channel dim in image (C_IN)
    // kLb/Ub : Values representing bounds on numFilters dim in filters (C_OUT)
    // gId    : Value representing induction variable for the outer loop
    void lowerConvolution(Value result,
                          Value images,
                          Value filters,
                          ArrayAttr stridesAttr,
                          ArrayAttr padBelowAttr,
                          ArrayAttr padAboveAttr,
                          PatternRewriter& rewriter,
                          DialectLoweringPass& pass,
                          Location loc,
                          Value cLb = nullptr,
                          Value cUb = nullptr,
                          Value kLb = nullptr,
                          Value kUb = nullptr,
                          Value gId = nullptr);

    template <typename OP>
    void lowerPooling(Operation* op,
                      ArrayRef<Value> operands,
                      PatternRewriter& rewriter,
                      DialectLoweringPass& pass);

    ValueHandle createZeroConstant(mlir::Type type);
    ValueHandle createOneConstant(mlir::Type type);

    /// Conversion from types in the nGraph dialect to the Standard dialect.
    class NGraphTypeConverter : public TypeConverter
    {
    public:
        NGraphTypeConverter()
            : TypeConverter()
        {
            // TODO(dcaballe): split this into independent conversion patterns when there is a
            // way to check if a type is valid in Std dialect.
            addConversion([this](Type type) -> Type {
                if (auto tensorType = type.dyn_cast<NGTensorType>())
                {
                    // Convert NGTensorType to Std MemRefType directly instead of going to Std
                    // TensorType. This may change in the future.
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

                // Do not assert/NGRAPH_CHECK here. Type convertion infra expects `convertType` to
                // return the input type if the type is not supported.
                return type;
            });
        }
    };

    /// Dialect Lowering Pass to affine ops
    class DialectLoweringPass : public ModulePass<DialectLoweringPass>
    {
    public:
        void runOnModule() override;

        SmallVector<Value, 4> buildOutputDefs(Operation* op, PatternRewriter& rewriter);
        /// Allocates a linear buffer for a temporary memref that shares its
        /// underlying memory. Used in conjunction with createTempMemref
        Value createTempBuffer(int bufferId, PatternRewriter& rewriter);
        /// Creates an allocation or view of a memref.
        /// type     MemRef Type
        /// buffer   Optional buffer value to create view over
        /// offset   Optional offset into the buffer this view starts at
        ///
        /// If buffer is null it allocates a Memref directly and Offset is ignored.
        /// If not, it creates a view over the pre-allocated buffer at the given offset.
        Value createTempMemref(Type type, Value buffer, unsigned offset, PatternRewriter& rewriter);
        /// Inserts dealloc Ops for each temporary allocated by AllocOp
        void insertDeallocs(PatternRewriter& rewriter);
        NGraphTypeConverter& getTypeConverter() { return typeConverter; }
        FuncOp getCallDecl(StringRef name,
                           ArrayRef<Type> args,
                           ArrayRef<Type> output,
                           PatternRewriter& rewriter);

        inline size_t insertAttrs(opAttrs attrs);

        MemoryAnalysis* getMemAnalysis() const { return m_memAnalysis; }
    private:
        /// Collect a set of patterns to convert from the nGraph dialect to Affine dialect.
        void populateNGraphToAffineConversionPatterns(OwningRewritePatternList& patterns);
        void findOutputValues();
        void insertNoAliasArgAttrs();

    private:
        NGraphTypeConverter typeConverter;
        // List of temporary memrefs to deallocate at end of function
        SmallVector<Value, 4> memRefsToDealloc;

        // Ops maybe assigned mem-refs in previous memory optimization passes.
        // Track pre-assigned buffers  for each Value and re-use it if one is available.
        using IdToMemRefMap = std::unordered_map<unsigned, Value>;
        IdToMemRefMap m_id_to_memref;
        MemoryAnalysis* m_memAnalysis;
        // TODO: Workaround for findOutputValues and buildOutputDefs. See NGCPU-470.
        StringRef funcName;

        // Store the attributes needed by callback
        std::vector<opAttrs> m_attrsVec;
    };

    void DialectLoweringPass::runOnModule()
    {
        // Create type converter and initialize conversion patterns.
        NGraphTypeConverter converter;
        OwningRewritePatternList patterns;

        populateNGraphToAffineConversionPatterns(patterns);

        // Get Memory analysis for in-place memory optimizations
        m_memAnalysis = &getAnalysis<MemoryAnalysis>();

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
            if (clEnableBarePtrMemRefLowering)
            {
                insertNoAliasArgAttrs();
            }
        }

        opAttrsVec = m_attrsVec;
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
        NGRAPH_CHECK(f, "FuncOp '" + funcName.str() + "' not found");

        SmallVector<Value, 4> outputList;
        unsigned outputCount = 0;
        unsigned inputCount = f.getType().getNumInputs();
        // we find out output values by looking at returned values
        // any return should return all outputs of the subgraph
        f.walk([&outputCount, inputCount](NGReturnOp ret) {
            for (unsigned i = 0; i < ret.getNumOperands(); i++)
            {
                // annotate instructions defining outputs with the arg idx of the output
                auto outputValue = ret.getOperand(i);
                auto op = outputValue.getDefiningOp();

                op->setAttr(
                    "graphOutputIdx",
                    mlir::IntegerAttr::get(IntegerType::get(32, op->getContext()), i + inputCount));
            }
            NGRAPH_CHECK(outputCount == 0 || outputCount == ret.getNumOperands(),
                         "Inconsistent returns in function");
        });
    }

    SmallVector<Value, 4> DialectLoweringPass::buildOutputDefs(Operation* op,
                                                               PatternRewriter& rewriter)
    {
        FuncOp f = getModule().lookupSymbol<mlir::FuncOp>(funcName);
        NGRAPH_CHECK(f, "FuncOp '" + funcName.str() + "' not found");

        SmallVector<Value, 4> newResults;
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
                // For temporaries, we create two instructions:
                // 1. Linear buffer allocation: If the ng value already has a buffer ID assigned,
                //    we re-use that linear buffer SSA value, else generate an AllocOp.
                // 2. View creation: Create a view with the tensor shape and an N-D to 1 map over
                //    the linear buffer.
                // If two memrefs are defined via 2 Views over the same buffer, then they share and
                // will re-use the same buffer.
                auto tensorType = origResult.getType().cast<NGTensorType>();
                Value newResult = nullptr;
                auto bufferInfo = m_memAnalysis->getBufferInfo(op);
                Type memRefType = typeConverter.convertType(tensorType);
                Value bufferValue = nullptr;

                if (!bufferInfo.isValid())
                {
                    // Allocate new memref
                    newResult = createTempMemref(memRefType, nullptr, 0, rewriter);
                }
                else
                {
                    unsigned bufferId = bufferInfo.m_bufferId;
                    unsigned offset = bufferInfo.m_offset;
                    // Re-use a buffer if it exist, else create a new one and update map
                    IdToMemRefMap::iterator it = m_id_to_memref.find(bufferId);
                    if (it == m_id_to_memref.end())
                    {
                        // create a new buffer
                        bufferValue = createTempBuffer(bufferId, rewriter);
                        m_id_to_memref[bufferId] = bufferValue;
                    }
                    else
                    {
                        bufferValue = it->second;
                    }
                    // Create a temp view over the linear buffer
                    newResult = createTempMemref(memRefType, bufferValue, offset, rewriter);
                }
                NGRAPH_CHECK(newResult != nullptr, "Temp memref value is not set");
                newResults.push_back(newResult);
            }
        }
        return newResults;
    }

    Value DialectLoweringPass::createTempBuffer(int bufferId, PatternRewriter& rewriter)
    {
        unsigned sizeInBytes = getMemAnalysis()->getBufferSize(bufferId);
        NGRAPH_CHECK(bufferId >= 0, "Invalid buffer id to allocate");
        NGRAPH_CHECK(sizeInBytes > 0, "Zero buffer allocation?");

        LLVM_DEBUG(llvm::dbgs() << "Allocating buffer of size " << sizeInBytes << " bytes\n");
        MemRefType bufferType =
            MemRefType::get({sizeInBytes}, IntegerType::get(8, rewriter.getContext()), {});

        // TODO: Set alignment
        Value alloc = rewriter.create<mlir::AllocOp>(rewriter.getUnknownLoc(), bufferType);

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

    Value DialectLoweringPass::createTempMemref(Type type,
                                                Value buffer,
                                                unsigned offset,
                                                PatternRewriter& rewriter)
    {
        MemRefType memRefType = type.cast<MemRefType>();
        if (buffer)
        {
            // We have a buffer to map to. Create a view over it.

            // Create the N-D to 1D affine expression mapping the memref shape to the underlying
            // linear
            // buffer
            // This is simply (d0, d1, d2, .. dN-1) --> d0 * S0 + d1 * S1 ... + dN-1 * SN-1
            // Where Si is the stride along the i_th dimension in elements
            auto shape = memRefType.getShape();
            SmallVector<int64_t, 4> strides(shape.size(), 0);
            strides[shape.size() - 1] = 1;
            for (int64_t i = shape.size() - 2; i >= 0; i--)
            {
                strides[i] = strides[i + 1] * shape[i + 1];
            }

            auto map = makeStridedLinearLayoutMap(strides, offset, rewriter.getContext());
            MemRefType newMemRefType = MemRefType::get(shape, memRefType.getElementType(), map);
            auto viewOp = rewriter.create<mlir::ViewOp>(
                buffer.getDefiningOp()->getLoc(), newMemRefType, buffer, llvm::None);
            return viewOp.getResult();
        }

        // No buffer, create an atomic memref without underlying buffer
        NGRAPH_CHECK(memRefType.hasStaticShape(), "Dynamic shapes are not supported");

        Value alloc = rewriter.create<mlir::AllocOp>(rewriter.getUnknownLoc(), memRefType);
        memRefsToDealloc.push_back(alloc);
        return alloc;
    }

    /// Add llvm.noalias attribute to all the memref function arguments. We know that this is safe
    /// by nGraph op semantics.
    void DialectLoweringPass::insertNoAliasArgAttrs()
    {
        FuncOp func = getModule().lookupSymbol<mlir::FuncOp>(funcName);
        NGRAPH_CHECK(func, "FuncOp '" + funcName.str() + "' not found");

        unsigned int argIdx = 0;
        for (auto arg : func.getArguments())
        {
            if (arg.getType().isa<MemRefType>())
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

    mlir::FuncOp DialectLoweringPass::getCallDecl(StringRef name,
                                                  ArrayRef<Type> args,
                                                  ArrayRef<Type> output,
                                                  PatternRewriter& rewriter)
    {
        auto module = getModule();
        auto callBackFunc = module.lookupSymbol<mlir::FuncOp>(name);
        if (!callBackFunc)
        {
            // Create a function declaration and insert to the module.
            auto callBackType = rewriter.getFunctionType(args, output);
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            SmallVector<NamedAttribute, 4> attributes;
            rewriter.create<mlir::FuncOp>(rewriter.getUnknownLoc(), name, callBackType, attributes);
            callBackFunc = module.lookupSymbol<mlir::FuncOp>(name);
        }
        return callBackFunc;
    }

    inline size_t DialectLoweringPass::insertAttrs(opAttrs attrs)
    {
        m_attrsVec.push_back(attrs);
        return m_attrsVec.size() - 1;
    }

#define REWRITER(OP)                                                                               \
    PatternMatchResult OP##Conversion::matchAndRewrite(                                            \
        Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const

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

    REWRITER(NGGreaterEqOp)
    {
        lowerBinaryElementwise<mlir::NGGreaterEqOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGLessEqOp)
    {
        lowerBinaryElementwise<mlir::NGLessEqOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGEqOp)
    {
        lowerBinaryElementwise<mlir::NGEqOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGNotEqOp)
    {
        lowerBinaryElementwise<mlir::NGNotEqOp>(op, operands, rewriter, pass);
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
        NGRAPH_CHECK(result.getType().isa<MemRefType>());
        // Note that builder's current function is still the original function body.
        // use getBlock to get the new block instead.

        // get new operands
        Value lhs = operands[0];

        ScopedContext scope(rewriter, loc);
        // Views
        MemRefBoundsCapture vRes(result), vLHS(lhs);
        // Index Values
        AffineIndexedValue iRes(result), iLHS(lhs);
        // Bounds Index Handles
        auto lbs = vLHS.getLbs();
        auto ubs = vLHS.getUbs();
        // Loop induction vars
        auto ivs = ValueHandle::makeIndexHandles(vLHS.rank());
        auto pivs = makeHandlePointers(ivs);
        // Steps
        auto steps = vLHS.getSteps();

        NGRAPH_CHECK(lhs.getType().isa<MemRefType>());
        Type elemTy = lhs.getType().dyn_cast<MemRefType>().getElementType();

        AffineLoopNestBuilder(pivs, lbs, ubs, steps)([&] {
            ValueHandle val = iLHS(ivs);
            ValueHandle zero = createZeroConstant(elemTy);
            iRes(ivs) = std_select(val > zero, val, zero);
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
        Value lhs = operands[0];
        Value rhs = operands[1];
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs && rhs && result, "Unexpected null values in DotOp");

        auto resultTy = result.getType().dyn_cast<MemRefType>();
        auto lhsTy = lhs.getType().dyn_cast<MemRefType>();
        auto rhsTy = rhs.getType().dyn_cast<MemRefType>();
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

        MemRefBoundsCapture vRes(result), vLhs(lhs), vRhs(rhs);

        NGRAPH_CHECK(vLhs.rank() == 2 && vRhs.rank() == 2 && vRes.rank() == 2,
                     "Dot operation is only supported for 2D tensors");

        // Create induction variables, lower bounds, upper bounds and steps of the loop nest.
        // It's important to note that MemRefBoundsCapture priovides lb/ub/step info is "reverse
        // order", i.e., fastest varying dimension is the last one, slowest varying dimention is the
        // first one.
        auto indexType = IndexType::get(rewriter.getContext());
        ValueHandle n(indexType), m(indexType), k(indexType);
        unsigned nDim = vLhs.fastestVarying() - 1;
        unsigned mDim = vRhs.fastestVarying();
        unsigned kDim = vRhs.fastestVarying();
        ValueHandle nLb(vLhs.lb(nDim)), mLb(vLhs.lb(mDim)), kLb(vRhs.lb(kDim));
        ValueHandle nUb(vLhs.ub(nDim)), mUb(vLhs.ub(mDim)), kUb(vRhs.ub(kDim));
        int64_t nStep = vLhs.step(nDim), mStep = vLhs.step(mDim), kStep = vRhs.step(kDim);

        // Constants and indexed values to be used inside the loop nest.
        AffineIndexedValue iRes(result), iLhs(lhs), iRhs(rhs);
        ValueHandle zeroInit(rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elemTy)));

        {
            ValueHandle n(indexType), k(indexType);
            makeAffineLoopBuilder(&n, nLb, nUb, nStep)([&] {
                makeAffineLoopBuilder(&k, kLb, kUb, kStep)([&] { iRes(n, k) = zeroInit; });
            });
        }
        makeAffineLoopBuilder(&n, nLb, nUb, nStep)([&] {
            makeAffineLoopBuilder(&m, mLb, mUb, mStep)([&] {
                makeAffineLoopBuilder(&k, kLb, kUb, kStep)(
                    [&] { iRes(n, k) += iLhs(n, m) * iRhs(m, k); });
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
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in ConcatOp");

        // Create view to write into result.
        MemRefBoundsCapture vRes(result);
        auto rank = vRes.rank();

        // For each operand, generate a separate loop to copy into the target slice of "result".
        // We'll keep track of the slice offsets via concatenation_axis_pos.
        auto concatenationAxis = concat.concatenation_axis().getSExtValue();
        Value concatenationAxisPos(std_constant_index(0));

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
            MemRefBoundsCapture vOperand(operand);
            NGRAPH_CHECK(vOperand.rank() == rank, "Unexpected rank mismatch");

            llvm::SmallVector<ValueHandle, 5> indexVars;
            llvm::SmallVector<ValueHandle*, 5> indexVarPtrs;
            llvm::SmallVector<ValueHandle, 5> indexVarLbs;
            llvm::SmallVector<ValueHandle, 5> indexVarUbs;
            llvm::SmallVector<int64_t, 5> indexVarSteps;
            auto indexType = IndexType::get(rewriter.getContext());
            for (int i = 0; i < rank; i++)
            {
                indexVars.push_back(ValueHandle(indexType));
                indexVarPtrs.push_back(&(indexVars.back()));
                indexVarLbs.push_back(vOperand.lb(i));
                indexVarUbs.push_back(vOperand.ub(i));
                indexVarSteps.push_back(vOperand.step(i));
            }

            AffineLoopNestBuilder(indexVarPtrs, indexVarLbs, indexVarUbs, indexVarSteps)([&] {
                AffineIndexedValue ivRes(result);
                AffineIndexedValue ivOperand(operand);

                // On the LHS of the assignment, adjust the index for the concatenation axis.
                llvm::SmallVector<ValueHandle, 5> resIndexHandles;
                for (int i = 0; i < rank; i++)
                {
                    resIndexHandles.push_back(i == concatenationAxis
                                                  ? indexVars[i] + ValueHandle(concatenationAxisPos)
                                                  : indexVars[i]);
                }

                ivRes(resIndexHandles) = ivOperand(indexVars);
            });

            // Move up concatenation_axis_pos for the next operand.
            concatenationAxisPos =
                ValueHandle(concatenationAxisPos) + vOperand.ub(concatenationAxis);
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
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in GatherOp");

        Value params = operands[0];
        Value indices = operands[1];
        auto axis = gatherOp.axis().getSExtValue();

        // Create view to write into result.
        MemRefBoundsCapture vRes(result), vParams(params), vIndices(indices);
        // Indexed Values
        AffineIndexedValue iRes(result), iIndices(indices);
        StdIndexedValue iParams(params);

        // Construct outer loop for params dims. Exclude the axis dim.
        SmallVector<ValueHandle, 4> paramsLbs, paramsUbs, paramsIVs;
        SmallVector<int64_t, 4> paramsSteps;
        SmallVector<ValueHandle*, 4> paramsIVPtrs;
        for (auto i = 0; i < vParams.rank(); i++)
        {
            // skip gather axis
            if (i == axis)
                continue;
            paramsLbs.push_back(vParams.lb(i));
            paramsUbs.push_back(vParams.ub(i));
            paramsSteps.push_back(vParams.step(i));
        }
        NGRAPH_CHECK(paramsLbs.size() == vParams.rank() - 1 &&
                         paramsUbs.size() == paramsLbs.size() &&
                         paramsSteps.size() == paramsLbs.size(),
                     "Incorrect loop nest bounds size for gather params");

        paramsIVs = ValueHandle::makeIndexHandles(vParams.rank() - 1);
        paramsIVPtrs = makeHandlePointers(paramsIVs);

        auto indicesLbs = vIndices.getLbs();
        auto indicesUbs = vIndices.getUbs();
        auto indicesSteps = vIndices.getSteps();

        auto indicesIVs = ValueHandle::makeIndexHandles(vIndices.rank());
        auto indicesIVPtrs = makeHandlePointers(indicesIVs);

        SmallVector<ValueHandle, 8> paramsIndices, resIndices;

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

        AffineLoopNestBuilder(indicesIVPtrs, indicesLbs, indicesUbs, indicesSteps)([&] {
            // Load axis value from indices array and cast it to Index Type
            ValueHandle axisIdx = ValueHandle::create<IndexCastOp>(
                (ValueHandle)iIndices(indicesIVs), rewriter.getIndexType());

            AffineLoopNestBuilder(paramsIVPtrs, paramsLbs, paramsUbs, paramsSteps)([&] {
                // construct indices for param
                // [P_0, P_1, .. P_axis-1, Indices[I0, I1, .. I_k-1], P_axis+1, P_axis+2, .. P_n-1]
                for (auto i = 0, j = 0; i < vParams.rank(); i++)
                {
                    if (i == axis)
                    {
                        paramsIndices.push_back(axisIdx);
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

        // Get operands
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in Convolution Op");
        Value images = operands[0];
        Value filters = operands[1];
        auto strides = convolOp.strides();
        auto padBelow = convolOp.padBelow();
        auto padAbove = convolOp.padBelow();

        lowerConvolution(result,
                         images,
                         filters,
                         strides,
                         padBelow,
                         padAbove,
                         rewriter,
                         pass,
                         convolOp.getLoc());

        rewriter.replaceOp(op, {result});
        return matchSuccess();
    }

    REWRITER(NGGroupConvOp)
    {
        auto gConvOp = cast<NGGroupConvOp>(op);
        ScopedContext scope(rewriter, gConvOp.getLoc());
        // Get operands
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in Convolution Op");
        Value images = operands[0];
        Value filters = operands[1];
        auto strides = gConvOp.strides();
        auto padBelow = gConvOp.padBelow();
        auto padAbove = gConvOp.padBelow();
        int groups = gConvOp.groups().getSExtValue();

        NGRAPH_CHECK(groups > 0, "Invalid number of groups");
        // create outer group convolution loop
        // for group = 0 to groups
        auto indexType = IndexType::get(rewriter.getContext());
        ValueHandle iv(indexType);
        ValueHandle lb = std_constant_index(0);
        ValueHandle ub = std_constant_index(groups);

        auto imagesType = images.getType().cast<MemRefType>();
        auto filtersType = filters.getType().cast<MemRefType>();
        auto imagesShape = imagesType.getShape();
        auto filtersShape = filtersType.getShape();

        // Filters shape contains num of groups ?
        bool groupsInFilters = (filtersShape.size() != imagesShape.size());

        NGRAPH_CHECK(imagesType.hasStaticShape() && filtersType.hasStaticShape(),
                     "Dynamic shapes are not supported");
        NGRAPH_CHECK(imagesShape[1] % groups == 0,
                     "Channel dim is not divisible by number of groups");

        NGRAPH_CHECK(groupsInFilters || filtersShape[0] % groups == 0,
                     "Filters dim is not divisible by number of groups");

        auto channelGroupSize = std_constant_index(imagesShape[1] / groups);
        auto filtersGroupSize =
            std_constant_index(groupsInFilters ? filtersShape[1] : filtersShape[0] / groups);

        NGRAPH_CHECK(!groupsInFilters || groups == filtersShape[0]);

        makeAffineLoopBuilder(&iv, lb, ub, 1)([&] {
            // lower/upper bounds on image channel dim and kernels dim
            auto cLb = iv * channelGroupSize;
            auto cUb = cLb + channelGroupSize;
            auto kLb = iv * filtersGroupSize;
            auto kUb = kLb + filtersGroupSize;
            lowerConvolution(result,
                             images,
                             filters,
                             strides,
                             padBelow,
                             padAbove,
                             rewriter,
                             pass,
                             gConvOp.getLoc(),
                             cLb,
                             cUb,
                             kLb,
                             kUb,
                             iv);
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

    // Use callback: Pooling, MatMul, Gemm, Softmax
    static void castMemRef(SmallVector<mlir::Value, 4>& inputs,
                           SmallVector<mlir::Value, 4>& outputs,
                           PatternRewriter& rewriter,
                           UnrankedMemRefType type)
    {
        for (auto in : inputs)
        {
            auto out = rewriter.create<mlir::MemRefCastOp>(rewriter.getUnknownLoc(), in, type);
            outputs.push_back(out);
        }
    }

    REWRITER(NGAvgPoolOp)
    {
        lowerPooling<mlir::NGAvgPoolOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGAvgPoolBackpropOp)
    {
        lowerPooling<mlir::NGAvgPoolBackpropOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGMaxPoolOp)
    {
        lowerPooling<mlir::NGMaxPoolOp>(op, operands, rewriter, pass);
        return matchSuccess();
    }

    REWRITER(NGMaxPoolBackpropOp)
    {
        auto pooling = cast<NGMaxPoolBackpropOp>(op);
        auto loc = pooling.getLoc();

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value src = operands[0];
        Value delta = operands[1];
        ArrayRef<Attribute> windowShape = pooling.windowShape().getValue();
        ArrayRef<Attribute> windowStrides = pooling.windowMovementStrides().getValue();
        ArrayRef<Attribute> padBelow = pooling.padBelow().getValue();
        ArrayRef<Attribute> padAbove = pooling.padAbove().getValue();

        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(src && delta && result, "Unexpected null values in MaxPoolBackprop Op");

        auto resultTy = result.getType().dyn_cast<MemRefType>();
        auto resultShape = resultTy.getShape();
        auto srcTy = src.getType().dyn_cast<MemRefType>();
        auto srcShape = srcTy.getShape();
        auto deltaTy = delta.getType().dyn_cast<MemRefType>();
        NGRAPH_CHECK(resultTy, "Unexpected non-memref result type");
        NGRAPH_CHECK(srcTy, "Unexpected non-memref src type");
        NGRAPH_CHECK(deltaTy, "Unexpected non-memref delta type");

        Type elemTy = resultTy.getElementType();
        NGRAPH_CHECK(elemTy == srcTy.getElementType() && elemTy == deltaTy.getElementType(),
                     "Types mismatch in MaxPoolBackprop");

        NGRAPH_CHECK((srcShape.size() == 4 && resultShape.size() == 4) ||
                         (srcShape.size() == 5 && resultShape.size() == 5),
                     "MKLDNN pooling operation is only supported for 3D and 5D tensors");

        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        SmallVector<mlir::Value, 4> inputs = {src, delta, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);

        FuncOp callBackFunc = pass.getCallDecl(
            "callback_2_inputs",
            {unrankedMemrefTy, unrankedMemrefTy, unrankedMemrefTy, int64Ty, int64Ty},
            {},
            rewriter);

        opAttrs attrs;
        if (srcShape.size() == 4)
        {
            attrs.poolAttrs2d.includePaddingInAvgComputation = false;
            for (auto i = 0; i < 2; i++)
            {
                attrs.poolAttrs2d.windowShape[i] = windowShape[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs2d.windowStrides[i] = windowStrides[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs2d.padBelow[i] = padBelow[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs2d.padAbove[i] = padAbove[i].cast<IntegerAttr>().getInt();
            }
        }
        else if (srcShape.size() == 5)
        {
            attrs.poolAttrs3d.includePaddingInAvgComputation = false;
            for (auto i = 0; i < 3; i++)
            {
                attrs.poolAttrs3d.windowShape[i] = windowShape[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs3d.windowStrides[i] = windowStrides[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs3d.padBelow[i] = padBelow[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs3d.padAbove[i] = padAbove[i].cast<IntegerAttr>().getInt();
            }
        }
        auto index = pass.insertAttrs(attrs);
        auto attrsIndexArg =
            rewriter.create<mlir::ConstantIntOp>(rewriter.getUnknownLoc(), index, 64);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::MAXPOOLBACKPROP), 64);
        SmallVector<mlir::Value, 4> args = {
            outputs[0], outputs[1], outputs[2], attrsIndexArg, opTypeArg};

        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);
        return matchSuccess();
    }

    REWRITER(NGMatMulOp)
    {
        auto matmul = cast<NGMatMulOp>(op);
        auto loc = matmul.getLoc();

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value lhs = operands[0];
        Value rhs = operands[1];
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs && rhs && result, "Unexpected null values in MatMulOp");

        auto resultTy = result.getType().dyn_cast<MemRefType>();
        auto resultShape = resultTy.getShape();
        auto lhsTy = lhs.getType().dyn_cast<MemRefType>();
        auto lhsShape = lhsTy.getShape();
        auto rhsTy = rhs.getType().dyn_cast<MemRefType>();
        auto rhsShape = rhsTy.getShape();
        NGRAPH_CHECK(resultTy, "Unexpected non-memref result type");
        NGRAPH_CHECK(lhsTy, "Unexpected non-memref LHS type");
        NGRAPH_CHECK(rhsTy, "Unexpected non-memref RHS type");

        Type elemTy = resultTy.getElementType();
        NGRAPH_CHECK(elemTy == lhsTy.getElementType() && elemTy == rhsTy.getElementType(),
                     "Types mismatch in MatMulOp");

        NGRAPH_CHECK(lhsShape.size() == 2 && rhsShape.size() == 2 && resultShape.size() == 2,
                     "MatMul operation is only supported for 2D tensors");

        opAttrs attrs;
        attrs.gemmAttrs2d.transposeA = matmul.transposeA();
        attrs.gemmAttrs2d.transposeB = matmul.transposeB();
        attrs.gemmAttrs2d.m = lhsShape[0];
        attrs.gemmAttrs2d.k = lhsShape[1];
        attrs.gemmAttrs2d.n = rhsShape[1];
        attrs.gemmAttrs2d.lda = lhsShape[1];
        attrs.gemmAttrs2d.ldb = rhsShape[1];

        if (matmul.transposeA())
        {
            attrs.gemmAttrs2d.m = lhsShape[1];
            attrs.gemmAttrs2d.k = lhsShape[0];
        }
        if (matmul.transposeB())
        {
            attrs.gemmAttrs2d.n = rhsShape[0];
        }
        attrs.gemmAttrs2d.ldc = attrs.gemmAttrs2d.n;

        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        auto callBackFunc = pass.getCallDecl(
            "callback_2_inputs",
            {unrankedMemrefTy, unrankedMemrefTy, unrankedMemrefTy, int64Ty, int64Ty},
            {},
            rewriter);

        auto index = pass.insertAttrs(attrs);
        auto attrsIndexArg =
            rewriter.create<mlir::ConstantIntOp>(rewriter.getUnknownLoc(), index, 64);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::MATMUL), 64);
        SmallVector<mlir::Value, 4> inputs = {lhs, rhs, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 4> args = {
            outputs[0], outputs[1], outputs[2], attrsIndexArg, opTypeArg};

        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);

        return matchSuccess();
    }

    REWRITER(NGGemmOp)
    {
        auto gemm = cast<NGGemmOp>(op);
        auto loc = gemm.getLoc();

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value lhs = operands[0];
        Value rhs = operands[1];
        Value bias = operands[2];
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs && rhs && bias && result, "Unexpected null values in GemmOp");

        auto resultTy = result.getType().dyn_cast<MemRefType>();
        auto lhsTy = lhs.getType().dyn_cast<MemRefType>();
        auto lhsShape = lhsTy.getShape();
        auto rhsTy = rhs.getType().dyn_cast<MemRefType>();
        auto rhsShape = rhsTy.getShape();
        auto biasTy = bias.getType().dyn_cast<MemRefType>();
        auto biasShape = biasTy.getShape();
        NGRAPH_CHECK(resultTy, "Unexpected non-memref result type");
        NGRAPH_CHECK(lhsTy, "Unexpected non-memref LHS type");
        NGRAPH_CHECK(rhsTy, "Unexpected non-memref RHS type");
        NGRAPH_CHECK(biasTy, "Unexpected non-memref bias type");

        Type elemTy = resultTy.getElementType();
        NGRAPH_CHECK(elemTy == lhsTy.getElementType() && elemTy == rhsTy.getElementType() &&
                         elemTy == biasTy.getElementType(),
                     "Types mismatch in GemmOp");

        MemRefBoundsCapture vRes(result), vLhs(lhs), vRhs(rhs), vBias(bias);

        NGRAPH_CHECK(vLhs.rank() == 2 && vRhs.rank() == 2 && vRes.rank() == 2 && vBias.rank() <= 2,
                     "Gemm operation is only supported for 2D tensors");

        opAttrs attrs;
        attrs.gemmAttrs2d.transposeA = gemm.transA();
        attrs.gemmAttrs2d.transposeB = gemm.transB();
        attrs.gemmAttrs2d.alpha = gemm.alpha().convertToFloat();
        attrs.gemmAttrs2d.beta = gemm.beta().convertToFloat();
        attrs.gemmAttrs2d.m = lhsShape[0];
        attrs.gemmAttrs2d.k = lhsShape[1];
        attrs.gemmAttrs2d.n = rhsShape[1];
        attrs.gemmAttrs2d.lda = lhsShape[1];
        attrs.gemmAttrs2d.ldb = rhsShape[1];

        if (gemm.transA())
        {
            attrs.gemmAttrs2d.m = lhsShape[1];
            attrs.gemmAttrs2d.k = lhsShape[0];
        }
        if (gemm.transB())
        {
            attrs.gemmAttrs2d.n = rhsShape[0];
        }
        attrs.gemmAttrs2d.ldc = attrs.gemmAttrs2d.n;

        BroadcastType broadcastHint = BroadcastType::ERROR;
        if (vBias.rank() == 0)
        {
            // Scalar
            broadcastHint = BroadcastType::ROWCOLUMN;
        }
        else if (vBias.rank() == 2)
        {
            if (biasShape[0] == attrs.gemmAttrs2d.m && biasShape[1] == 1)
            {
                broadcastHint = BroadcastType::COLUMN;
            }
            else if (biasShape[0] == 1 && biasShape[1] == attrs.gemmAttrs2d.n)
            {
                broadcastHint = BroadcastType::ROW;
            }
            else if (biasShape[0] == attrs.gemmAttrs2d.m && biasShape[1] == attrs.gemmAttrs2d.n)
            {
                broadcastHint = BroadcastType::NONE;
            }
        }
        else
        {
            if (biasShape[0] == attrs.gemmAttrs2d.m)
            {
                broadcastHint = BroadcastType::COLUMN;
            }
            else if (biasShape[0] == attrs.gemmAttrs2d.n)
            {
                broadcastHint = BroadcastType::ROW;
            }
        }
        NGRAPH_CHECK(broadcastHint != BroadcastType::ERROR, "Unhandled broadcast");
        attrs.gemmAttrs2d.broadcastHint = broadcastHint;

        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        auto callBackFunc = pass.getCallDecl("callback_3_inputs",
                                             {unrankedMemrefTy,
                                              unrankedMemrefTy,
                                              unrankedMemrefTy,
                                              unrankedMemrefTy,
                                              int64Ty,
                                              int64Ty},
                                             {},
                                             rewriter);

        auto index = pass.insertAttrs(attrs);
        auto attrsIndexArg =
            rewriter.create<mlir::ConstantIntOp>(rewriter.getUnknownLoc(), index, 64);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::GEMM), 64);
        SmallVector<mlir::Value, 4> inputs = {lhs, rhs, bias, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 4> args = {
            outputs[0], outputs[1], outputs[2], outputs[3], attrsIndexArg, opTypeArg};

        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);

        return matchSuccess();
    }

    REWRITER(NGSoftMaxOp)
    {
        auto softmax = cast<NGSoftMaxOp>(op);
        auto loc = softmax.getLoc();

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value lhs = operands[0];
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs && result, "Unexpected null values in SoftmaxOp");

        auto resultTy = result.getType().dyn_cast<MemRefType>();
        auto resultShape = resultTy.getShape();
        auto lhsTy = lhs.getType().dyn_cast<MemRefType>();
        auto lhsShape = lhsTy.getShape();
        NGRAPH_CHECK(resultTy, "Unexpected non-memref result type");
        NGRAPH_CHECK(lhsTy, "Unexpected non-memref LHS type");

        Type elemTy = resultTy.getElementType();
        NGRAPH_CHECK(elemTy == lhsTy.getElementType(), "Types mismatch in SoftmaxOp");

        NGRAPH_CHECK((lhsShape.size() == 2 && resultShape.size() == 2) ||
                         (lhsShape.size() == 4 && resultShape.size() == 4),
                     "MKLDNN Softmax operation is only supported for 2D and 4D tensors");

        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        auto axes = softmax.axes().getValue();
        opAttrs attrs;
        attrs.intAttr = axes[0].cast<IntegerAttr>().getInt();
        auto index = pass.insertAttrs(attrs);
        auto attrsIndexArg =
            rewriter.create<mlir::ConstantIntOp>(rewriter.getUnknownLoc(), index, 64);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::SOFTMAX), 64);

        FuncOp callBackFunc =
            pass.getCallDecl("callback_1_input",
                             {unrankedMemrefTy, unrankedMemrefTy, int64Ty, int64Ty},
                             {},
                             rewriter);

        SmallVector<mlir::Value, 4> inputs = {lhs, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 4> args = {outputs[0], outputs[1], attrsIndexArg, opTypeArg};

        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);

        return matchSuccess();
    }

#undef REWRITER
    /// End of pattern matchers

    void lowerConvolution(Value result,
                          Value images,
                          Value filters,
                          ArrayAttr stridesAttr,
                          ArrayAttr padBelowAttr,
                          ArrayAttr padAboveAttr,
                          PatternRewriter& rewriter,
                          DialectLoweringPass& pass,
                          Location loc,
                          Value cLb,
                          Value cUb,
                          Value kLb,
                          Value kUb,
                          Value gId)
    {
        // Let Images shape be  [N, C_IN, D_1, ... D_f]
        // Let Filters shape be [C_OUT, C_IN, F_1, ... F_f]
        //      (or [GROUPS, C_OUT, C_IN, F_1, ... F_f] in case of
        //       group convolution with groups in filters shape)
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

        ScopedContext scope(rewriter, loc);
        auto strides = stridesAttr.getValue();
        auto padBelow = padBelowAttr.getValue();
        auto padAbove = padBelowAttr.getValue();
        Type elemTy = images.getType().cast<MemRefType>().getElementType();
        auto indexType = IndexType::get(rewriter.getContext());

        // Create views
        MemRefBoundsCapture vRes(result), vImages(images), vFilters(filters);
        // Create indexed Values
        AffineIndexedValue iRes(result), iImages(images), iFilters(filters);
        // Bounds on batch size N
        ValueHandle batchLb = vImages.lb(0), batchUb = vImages.ub(0);
        // Bounds on spatial dimensions
        SmallVector<ValueHandle, 4> resSpatialLbs, resSpatialUbs;
        SmallVector<ValueHandle, 4> imgSpatialLbs, imgSpatialUbs;
        SmallVector<ValueHandle, 4> filtersSpatialLbs, filtersSpatialUbs;
        // Spatial rank
        unsigned spatialRank = vImages.rank() - 2;

        // Result spatial indices and bounds
        auto resSpatialIndices = ValueHandle::makeIndexHandles(spatialRank);
        auto resSpatialIndicesPtrs = makeHandlePointers(resSpatialIndices);
        SmallVector<int64_t, 4> resSteps, filtersSteps;
        SmallVector<int, 4> padBelowIntValues;
        bool withPadding = false;

        // Do we have an extra dim for groups or is it folded in numFilters ?
        bool groupsInFilters = (vImages.rank() != vFilters.rank());
        bool groupConvolution = (kLb != nullptr);

        // Number of groups can be in filters shape only with group convolution
        NGRAPH_CHECK(!groupsInFilters ||
                     (kLb != nullptr && kUb != nullptr && cLb != nullptr && cUb != nullptr));

        // Bounds on number of filters
        ValueHandle numFiltersLb(rewriter.getIndexType());
        ValueHandle numFiltersUb(rewriter.getIndexType());
        if (groupConvolution)
        {
            if (groupsInFilters)
            {
                // use entire dim size if groups are out of the num filters dim
                numFiltersLb = vFilters.lb(1);
                numFiltersUb = vFilters.ub(1);
            }
            else
            {
                // use split dim within bounds generated in outer loop
                numFiltersLb = ValueHandle(kLb);
                numFiltersUb = ValueHandle(kUb);
            }
        }
        else
        {
            numFiltersLb = vFilters.lb(0);
            numFiltersUb = vFilters.ub(0);
        }

        // determine where spatial index starts in filters
        int filtersSpatialIdx = 2;
        const int imgSpatialIdx = 2;
        if (groupConvolution && groupsInFilters)
        {
            filtersSpatialIdx = 3;
        }
        // Bounds on number of channels
        ValueHandle numChannelsLb = (cLb == nullptr) ? vImages.lb(1) : ValueHandle(cLb);
        ValueHandle numChannelsUb = (cUb == nullptr) ? vImages.ub(1) : ValueHandle(cUb);

        for (auto i = 0; i < spatialRank; i++)
        {
            // result spatial bounds and steps
            resSpatialLbs.push_back(vRes.lb(imgSpatialIdx + i));
            resSpatialUbs.push_back(vRes.ub(imgSpatialIdx + i));
            resSteps.push_back(vRes.step(imgSpatialIdx + i));
            // image spatial bounds
            imgSpatialLbs.push_back(vImages.lb(imgSpatialIdx + i));
            imgSpatialUbs.push_back(vImages.ub(imgSpatialIdx + i));

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

        NGRAPH_CHECK((groupConvolution && groupsInFilters) || (vImages.rank() == vFilters.rank()),
                     "Images and Filters have unequal ranks");
        NGRAPH_CHECK(resSpatialLbs.size() == resSpatialUbs.size() &&
                         resSpatialLbs.size() == spatialRank,
                     "Results spatial dims mismatches input");

        // Filters spatial indices and bounds
        auto filtersSpatialIndices = ValueHandle::makeIndexHandles(spatialRank);
        auto filtersSpatialIndicesPtrs = makeHandlePointers(filtersSpatialIndices);

        for (auto i = 0; i < spatialRank; i++)
        {
            filtersSpatialLbs.push_back(vFilters.lb(filtersSpatialIdx + i));
            filtersSpatialUbs.push_back(vFilters.ub(filtersSpatialIdx + i));
            filtersSteps.push_back(vFilters.step(filtersSpatialIdx + i));
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
            nonPaddedRange = IntegerSet::get(spatialRank, 2 * spatialRank, affineExprs, isEq);
        }

        // Initialize output to zero
        {
            ValueHandle n(indexType), k(indexType), c(indexType);
            auto resSpatialIndices = ValueHandle::makeIndexHandles(spatialRank);
            auto resSpatialIndicesPtrs = makeHandlePointers(resSpatialIndices);

            makeAffineLoopBuilder(&n, batchLb, batchUb, 1)([&] {
                makeAffineLoopBuilder(&k, numFiltersLb, numFiltersUb, 1)([&] {
                    AffineLoopNestBuilder(
                        resSpatialIndicesPtrs, resSpatialLbs, resSpatialUbs, resSteps)([&] {
                        SmallVector<ValueHandle, 4> resIndices;
                        // Result indices
                        resIndices.push_back(n);
                        if (groupConvolution && groupsInFilters)
                        {
                            // compute global C_OUT from gID and k
                            // gId * C_OUT (num of filters) + k
                            resIndices.push_back(ValueHandle(gId) * numFiltersUb + k);
                        }
                        else
                        {
                            resIndices.push_back(k);
                        }
                        resIndices.insert(
                            resIndices.end(), resSpatialIndices.begin(), resSpatialIndices.end());
                        ValueHandle zero = createZeroConstant(elemTy);
                        iRes(resIndices) = zero;
                    });
                });
            });
        }

        ValueHandle n(indexType), k(indexType), c(indexType);
        // Convolution loop
        makeAffineLoopBuilder(&n, batchLb, batchUb, 1)([&] {
            // Number of filters loop
            makeAffineLoopBuilder(&k, numFiltersLb, numFiltersUb, 1)([&] {
                // Channels loop
                makeAffineLoopBuilder(&c, numChannelsLb, numChannelsUb, 1)([&] {
                    // Results loop
                    AffineLoopNestBuilder(
                        resSpatialIndicesPtrs, resSpatialLbs, resSpatialUbs, resSteps)([&] {
                        // Compute image start indices
                        SmallVector<ValueHandle, 4> imgStartIndices;
                        for (auto i = 0; i < spatialRank; i++)
                        {
                            IntegerAttr iAttr = strides[i].cast<IntegerAttr>();
                            auto stride = std_constant_index(iAttr.getInt());
                            imgStartIndices.push_back(resSpatialIndices[i] * stride);
                        }
                        SmallVector<ValueHandle, 4> resIndices;
                        // Result indices
                        resIndices.push_back(n);
                        if (groupConvolution && groupsInFilters)
                        {
                            // gId * C_OUT (num of filters) + k
                            resIndices.push_back(ValueHandle(gId) * numFiltersUb + k);
                        }
                        else
                        {
                            resIndices.push_back(k);
                        }

                        resIndices.insert(
                            resIndices.end(), resSpatialIndices.begin(), resSpatialIndices.end());
                        // Filters spatial loop
                        AffineLoopNestBuilder(filtersSpatialIndicesPtrs,
                                              filtersSpatialLbs,
                                              filtersSpatialUbs,
                                              filtersSteps)([&] {
                            SmallVector<ValueHandle, 4> imgIndices, filtersIndices;
                            // Image indices
                            // Here we compute the virtual start index into the padded image.
                            imgIndices.push_back(n);
                            imgIndices.push_back(c);
                            for (auto i = 0; i < spatialRank; i++)
                            {
                                imgIndices.push_back(imgStartIndices[i] + filtersSpatialIndices[i]);
                            }
                            // Filter indices

                            // If we are doing group convolution and filters shape dim0
                            // holds the number of groups, we need to use group id as the first
                            // index
                            if (groupConvolution && groupsInFilters)
                            {
                                filtersIndices.push_back(ValueHandle(gId));
                            }

                            filtersIndices.push_back(k);
                            // subtract lower bound of channel
                            // if we are doing group convolution this bound will advance based
                            // on the group id. For the filters, it should always start from 0
                            filtersIndices.push_back(c - numChannelsLb);
                            filtersIndices.insert(filtersIndices.end(),
                                                  filtersSpatialIndices.begin(),
                                                  filtersSpatialIndices.end());

                            if (withPadding)
                            {
                                // if args : img dims, img lbs, img ubs
                                SmallVector<ValueHandle, 4>::iterator it = imgIndices.begin();
                                std::advance(it, 2);
                                SmallVector<Value, 4> affineIfArgs(it, imgIndices.end());
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
                                    SmallVector<ValueHandle, 4> adjustedImgIndices;
                                    adjustedImgIndices.push_back(n);
                                    adjustedImgIndices.push_back(c);
                                    for (auto i = 0; i < spatialRank; i++)
                                    {
                                        adjustedImgIndices.push_back(
                                            imgIndices[2 + i] -
                                            std_constant_index(padBelowIntValues[i]));
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
    }

    template <typename OP>
    void lowerUnaryElementwise(Operation* op,
                               ArrayRef<Value> operands,
                               PatternRewriter& rewriter,
                               DialectLoweringPass& pass)
    {
        auto loc = cast<OP>(op).getLoc();

        auto result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result.getType().isa<MemRefType>());
        // Note that builder's current function is still the original function body.
        // use getBlock to get the new block instead.

        // get new operands
        Value lhs = operands[0];

        ScopedContext scope(rewriter, loc);
        // Views
        MemRefBoundsCapture vRes(result), vLHS(lhs);
        // Index Values
        AffineIndexedValue iRes(result), iLHS(lhs);
        // Bounds Index Handles
        auto lbs = vLHS.getLbs();
        auto ubs = vLHS.getUbs();
        // Loop induction vars
        auto ivs = ValueHandle::makeIndexHandles(vLHS.rank());
        auto pivs = makeHandlePointers(ivs);
        // Steps
        auto steps = vLHS.getSteps();

        NGRAPH_CHECK(lhs.getType().isa<MemRefType>());
        Type elemTy = lhs.getType().cast<MemRefType>().getElementType();

        AffineLoopNestBuilder(pivs, lbs, ubs, steps)([&] {
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
                                ArrayRef<Value> operands,
                                PatternRewriter& rewriter,
                                DialectLoweringPass& pass)
    {
        auto loc = cast<OP>(op).getLoc();
        auto result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result.getType().isa<MemRefType>());
        // get new operands
        Value lhs = operands[0];
        Value rhs = operands[1];

        ScopedContext scope(rewriter, loc);
        // Views
        MemRefBoundsCapture vRes(result), vLHS(lhs), vRHS(rhs);
        // Index Values
        AffineIndexedValue iRes(result), iLHS(lhs), iRHS(rhs);
        // Bounds Index Handles
        auto lbs = vLHS.getLbs();
        auto ubs = vLHS.getUbs();
        // Loop induction vars
        auto ivs = ValueHandle::makeIndexHandles(vLHS.rank());
        auto pivs = makeHandlePointers(ivs);
        // Steps
        auto steps = vLHS.getSteps();
        // element type of the operand
        Type elemTy = result.getType().cast<MemRefType>().getElementType();
        AffineLoopNestBuilder(pivs, lbs, ubs, steps)(
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
                // TODO(pthoreho) For all comparision operators, use
                // zero_extendi(ValueHandle(iLHS(ivs)) !=
                // ValueHandle(iRHS(ivs)), IntegerType::get(8, op->getContext()));
                // instead of std_select once `zero_extendi` is
                // made available in the edsc::intrinsics namescope in MLIR repo.
                else if (isa<NGGreaterOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) > ValueHandle(iRHS(ivs)),
                                           createOneConstant(elemTy),
                                           createZeroConstant(elemTy));
                }
                else if (isa<NGLessOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) < ValueHandle(iRHS(ivs)),
                                           createOneConstant(elemTy),
                                           createZeroConstant(elemTy));
                }
                else if (isa<NGGreaterEqOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) >= ValueHandle(iRHS(ivs)),
                                           createOneConstant(elemTy),
                                           createZeroConstant(elemTy));
                }
                else if (isa<NGLessEqOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) <= ValueHandle(iRHS(ivs)),
                                           createOneConstant(elemTy),
                                           createZeroConstant(elemTy));
                }
                else if (isa<NGEqOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) == ValueHandle(iRHS(ivs)),
                                           createOneConstant(elemTy),
                                           createZeroConstant(elemTy));
                }
                else if (isa<NGNotEqOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) != ValueHandle(iRHS(ivs)),
                                           createOneConstant(elemTy),
                                           createZeroConstant(elemTy));
                }
                else if (isa<NGMaxOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) > ValueHandle(iRHS(ivs)),
                                           ValueHandle(iLHS(ivs)),
                                           ValueHandle(iRHS(ivs)));
                }
                else if (isa<NGMinOp>(op))
                {
                    iRes(ivs) = std_select(ValueHandle(iLHS(ivs)) < ValueHandle(iRHS(ivs)),
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
                             ArrayRef<Value> operands,
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
        Value arg = operands[0];

        Value result = pass.buildOutputDefs(op, rewriter)[0];

        // Views
        MemRefBoundsCapture vRes(result), vArg(arg);
        // Index Values
        StdIndexedValue iRes(result), stdArg(arg);
        AffineIndexedValue affineArg(arg);
        // Bounds Index Handles
        auto resLbs = vRes.getLbs();
        auto resUbs = vRes.getUbs();
        auto argLbs = vArg.getLbs();
        auto argUbs = vArg.getUbs();

        Type resTy = result.getType().cast<MemRefType>().getElementType();
        // Generate loop nest that initializes result to lower bound of the axis to be reduced.
        {
            auto ivs = ValueHandle::makeIndexHandles(vRes.rank());
            auto pivs = makeHandlePointers(ivs);
            auto steps = vRes.getSteps();
            auto initVal = vArg.lb(axis);
            AffineLoopNestBuilder(pivs, resLbs, resUbs, steps)(
                [&] { iRes(ivs) = ValueHandle::create<IndexCastOp>(initVal, resTy); });
        }

        // Generate loop nest that computes the actual index reduction.
        {
            auto allIVs = ValueHandle::makeIndexHandles(vArg.rank());
            auto pAllIVs = makeHandlePointers(allIVs);
            auto steps = vArg.getSteps();
            SmallVector<ValueHandle, 8> nonRedIVs;

            Type resTy = result.getType().cast<MemRefType>().getElementType();
            NGRAPH_CHECK(resTy.isa<IntegerType>(),
                         "Expected integer result type in index reduction");

            // iterate over all argument dimensions
            AffineLoopNestBuilder(pAllIVs, argLbs, argUbs, steps)([&] {
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
                        ? std_select(affineArg(allIVs) < stdArg(tempIVs), allIVs[axis], currRedIdx)
                        : std_select(stdArg(tempIVs) < affineArg(allIVs), allIVs[axis], currRedIdx);

                iRes(nonRedIVs) = ValueHandle::create<IndexCastOp>(newRedIdx, resTy);
            });
        }

        rewriter.replaceOp(op, result);
    }

    template <typename OP>
    void lowerPooling(Operation* op,
                      ArrayRef<Value> operands,
                      PatternRewriter& rewriter,
                      DialectLoweringPass& pass)
    {
        auto pooling = cast<OP>(op);
        auto loc = pooling.getLoc();

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value lhs = operands[0];
        ArrayRef<Attribute> windowShape = pooling.windowShape().getValue();
        ArrayRef<Attribute> windowStrides = pooling.windowMovementStrides().getValue();
        ArrayRef<Attribute> padBelow = pooling.padBelow().getValue();
        ArrayRef<Attribute> padAbove = pooling.padAbove().getValue();

        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs && result, "Unexpected null values in Pooling Op");

        auto resultTy = result.getType().dyn_cast<MemRefType>();
        auto resultShape = resultTy.getShape();
        auto lhsTy = lhs.getType().dyn_cast<MemRefType>();
        auto lhsShape = lhsTy.getShape();
        NGRAPH_CHECK(resultTy, "Unexpected non-memref result type");
        NGRAPH_CHECK(lhsTy, "Unexpected non-memref LHS type");

        Type elemTy = resultTy.getElementType();
        NGRAPH_CHECK(elemTy == lhsTy.getElementType(), "Types mismatch in Pooling");

        NGRAPH_CHECK((lhsShape.size() == 4 && resultShape.size() == 4) ||
                         (lhsShape.size() == 5 && resultShape.size() == 5),
                     "MKLDNN pooling operation is only supported for 3D and 5D tensors");

        auto int64Ty = rewriter.getIntegerType(64);
        OpType ty;
        bool includePadding = false;
        if (auto avgPool = dyn_cast<NGAvgPoolOp>(op))
        {
            ty = OpType::AVGPOOL;
            includePadding = avgPool.includePadding();
        }
        else if (auto avgPoolBprop = dyn_cast<NGAvgPoolBackpropOp>(op))
        {
            ty = OpType::AVGPOOLBACKPROP;
            includePadding = avgPoolBprop.includePadding();
        }
        else if (isa<NGMaxPoolOp>(op))
        {
            ty = OpType::MAXPOOL;
        }
        else
        {
            NGRAPH_UNREACHABLE("Unsupported pooling op");
        }

        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        SmallVector<mlir::Value, 4> inputs = {lhs, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);

        FuncOp callBackFunc =
            pass.getCallDecl("callback_1_input",
                             {unrankedMemrefTy, unrankedMemrefTy, int64Ty, int64Ty},
                             {},
                             rewriter);

        opAttrs attrs;
        if (lhsShape.size() == 4)
        {
            attrs.poolAttrs2d.includePaddingInAvgComputation = includePadding;
            for (auto i = 0; i < 2; i++)
            {
                attrs.poolAttrs2d.windowShape[i] = windowShape[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs2d.windowStrides[i] = windowStrides[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs2d.padBelow[i] = padBelow[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs2d.padAbove[i] = padAbove[i].cast<IntegerAttr>().getInt();
            }
        }
        else if (lhsShape.size() == 5)
        {
            attrs.poolAttrs3d.includePaddingInAvgComputation = includePadding;
            for (auto i = 0; i < 3; i++)
            {
                attrs.poolAttrs3d.windowShape[i] = windowShape[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs3d.windowStrides[i] = windowStrides[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs3d.padBelow[i] = padBelow[i].cast<IntegerAttr>().getInt();
                attrs.poolAttrs3d.padAbove[i] = padAbove[i].cast<IntegerAttr>().getInt();
            }
        }
        auto index = pass.insertAttrs(attrs);
        auto attrsIndexArg =
            rewriter.create<mlir::ConstantIntOp>(rewriter.getUnknownLoc(), index, 64);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(ty), 64);
        SmallVector<mlir::Value, 4> args = {outputs[0], outputs[1], attrsIndexArg, opTypeArg};

        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);
    }

    ValueHandle createZeroConstant(mlir::Type type)
    {
        if (auto floatTy = type.dyn_cast<FloatType>())
        {
            if (floatTy.isF32())
            {
                return std_constant_float(llvm::APFloat(0.0f), floatTy);
            }
            else if (floatTy.isF64())
            {
                return std_constant_float(llvm::APFloat(0.0), floatTy);
            }
            else
            {
                NGRAPH_UNREACHABLE("Unsupported floating-point precision");
            }
        }
        else if (auto intTy = type.dyn_cast<IntegerType>())
        {
            return std_constant_int(0, intTy.getWidth());
        }
        NGRAPH_UNREACHABLE("Unsupported type");
    }

    ValueHandle createOneConstant(mlir::Type type)
    {
        if (auto floatTy = type.dyn_cast<FloatType>())
        {
            if (floatTy.isF32())
            {
                return std_constant_float(llvm::APFloat(1.0f), floatTy);
            }
            else if (floatTy.isF64())
            {
                return std_constant_float(llvm::APFloat(1.0f), floatTy);
            }
            else
            {
                NGRAPH_UNREACHABLE("Unsupported floating-point precision");
            }
        }
        else if (auto intTy = type.dyn_cast<IntegerType>())
        {
            return std_constant_int(1, intTy.getWidth());
        }
        NGRAPH_UNREACHABLE("Unsupported type");
    }
} // namespace

namespace mlir
{
    std::unique_ptr<Pass> createDialectLoweringPass()
    {
        return std::make_unique<DialectLoweringPass>();
    }
} // namespace mlir

static PassRegistration<DialectLoweringPass> pass(PASS_NAME,
                                                  "Convert nGraph dialect to affine dialect");
