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

// NOTE: This file follows nGraph format style and MLIR naming convention since
// it does
// not expose public API to the rest of nGraph codebase and heavily depends on
// MLIR API.

#include "affine_lowerer.hpp"

#include "contrib/mlir/backend/analysis/memory_analysis.hpp"
#include "contrib/mlir/core/ngraph_dialect/ops.hpp"
#include "contrib/mlir/core/ngraph_dialect/type.hpp"
#include "contrib/mlir/runtime/cpu/callback_utils.hpp"
#include "contrib/mlir/utils.hpp"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "ngraph/assertion.hpp"

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Affine/EDSC/Builders.h>
#include <mlir/Dialect/Affine/EDSC/Intrinsics.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
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

    /// Base class for nGraph operation conversions to affine/standard dialect.
    /// Provides
    /// conversion patterns with an access to the DialectLoweringPass which holds
    /// the state of the
    /// conversion.
    class NGraphOpLowering : public ConversionPattern
    {
    public:
        NGraphOpLowering(StringRef rootOpName, MLIRContext* context, DialectLoweringPass& pass)
            : ConversionPattern(rootOpName, /*benefit=*/1, context)
            , pass(pass)
            , context(context){};

    protected:
        // Back-reference to the lowering pass which contains the lowering state,
        // including the
        // nGraph type converter.
        DialectLoweringPass& pass;
        MLIRContext* context;
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
        LogicalResult matchAndRewrite(Operation* op,                                               \
                                      ArrayRef<Value> operands,                                    \
                                      ConversionPatternRewriter& rewriter) const override;         \
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
        LogicalResult matchAndRewrite(Operation* op,
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
                    return failure();
                }
            }

            auto funcTypeResults = type.getResults();
            if (!funcTypeResults.empty())
            {
                // Convert the original function results.
                SmallVector<Type, 4> convertedResults;
                if (failed(converter.convertTypes(funcTypeResults, convertedResults)))
                {
                    return failure();
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
            if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), converter, &result)))
                return failure();
            rewriter.replaceOp(op, llvm::None);
            return success();
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
                      DialectLoweringPass& pass,
                      MLIRContext* context);

    Value createZeroConstant(mlir::Type type);
    Value createOneConstant(mlir::Type type);

    /// Conversion from types in the nGraph dialect to the Standard dialect.
    class NGraphTypeConverter : public TypeConverter
    {
    public:
        NGraphTypeConverter()
            : TypeConverter()
        {
            // TODO(dcaballe): split this into independent conversion patterns when
            // there is a
            // way to check if a type is valid in Std dialect.
            addConversion([this](Type type) -> Type {
                if (auto tensorType = type.dyn_cast<NGTensorType>())
                {
                    // Convert NGTensorType to Std MemRefType directly instead of going to
                    // Std
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

                // Do not assert/NGRAPH_CHECK here. Type convertion infra expects
                // `convertType` to
                // return the input type if the type is not supported.
                return type;
            });
        }
    };

    // Return llvm type for given attributes type
    static LLVM::LLVMType getLLVMType(AttrsType attrsType, MLIRContext* context)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(context);
        auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(context);
        auto llvmArray1DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 1);
        auto llvmArray2DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 2);
        auto llvmArray3DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 3);
        auto llvmF32Ty = LLVM::LLVMType::getFloatTy(context);
        switch (attrsType)
        {
        case AttrsType::INT: return llvmI64Ty;
        case AttrsType::CONV1D:
            return LLVM::LLVMType::getStructTy(
                context,
                {llvmI8Ty, llvmArray1DI64Ty, llvmArray1DI64Ty, llvmArray1DI64Ty, llvmArray1DI64Ty});
        case AttrsType::CONV2D:
            return LLVM::LLVMType::getStructTy(
                context,
                {llvmI8Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty});
        case AttrsType::CONV3D:
            return LLVM::LLVMType::getStructTy(
                context,
                {llvmI8Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty});
        case AttrsType::POOL2D:
            return LLVM::LLVMType::getStructTy(
                context,
                {llvmI8Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty});
        case AttrsType::POOL3D:
            return LLVM::LLVMType::getStructTy(
                context,
                {llvmI8Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty});
        case AttrsType::GEMM:
            return LLVM::LLVMType::getStructTy(context,
                                               {llvmI8Ty,
                                                llvmI8Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmF32Ty,
                                                llvmF32Ty,
                                                llvmI32Ty});
        }
    }

    // Create a Constant op and a Store op which stores the Constant
    static void
        createStore(LLVM::LLVMType llvmTy, Attribute valAttr, LLVM::GEPOp gep, OpBuilder& builder)
    {
        auto valueOp = builder.create<LLVM::ConstantOp>(builder.getUnknownLoc(), llvmTy, valAttr);
        builder.create<LLVM::StoreOp>(builder.getUnknownLoc(), valueOp, gep);
    }

    /// Dialect Lowering Pass to affine ops
    class DialectLoweringPass : public PassWrapper<DialectLoweringPass, OperationPass<ModuleOp>>
    {
    public:
        void runOnOperation() override;

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
        /// If not, it creates a view over the pre-allocated buffer at the given
        /// offset.
        Value createTempMemref(Type type, Value buffer, unsigned offset, PatternRewriter& rewriter);
        /// Inserts dealloc Ops for each temporary allocated by AllocOp
        void insertDeallocs(PatternRewriter& rewriter);
        NGraphTypeConverter& getTypeConverter() { return typeConverter; }
        FuncOp getCallDecl(StringRef name,
                           ArrayRef<Type> args,
                           ArrayRef<Type> output,
                           PatternRewriter& rewriter);

        // Return a GlobalOp with the given name
        // If such GlobalOp does not exist, create one
        mlir::LLVM::GlobalOp getGlobalOp(StringRef name,
                                         LLVM::LLVMType globalType,
                                         bool isConstant,
                                         LLVM::Linkage linkageType,
                                         Attribute initVal,
                                         OpBuilder& rewriter);

        /// Insert a function to the module which initializes the global variables
        /// that hold the attributes information for callbacks.
        void insertInitFunc();

        inline int32_t insertAttrs(opAttrs attrs, AttrsType type);

        MemoryAnalysis* getMemAnalysis() const { return m_memAnalysis; }

    private:
        /// Collect a set of patterns to convert from the nGraph dialect to Affine
        /// dialect.
        void populateNGraphToAffineConversionPatterns(OwningRewritePatternList& patterns);
        void findOutputValues();
        void insertNoAliasArgAttrs();

    private:
        NGraphTypeConverter typeConverter;
        // List of temporary memrefs to deallocate at end of function
        SmallVector<Value, 4> memRefsToDealloc;

        // Ops maybe assigned mem-refs in previous memory optimization passes.
        // Track pre-assigned buffers  for each Value and re-use it if one is
        // available.
        using IdToMemRefMap = std::unordered_map<unsigned, Value>;
        IdToMemRefMap m_id_to_memref;
        MemoryAnalysis* m_memAnalysis;
        // TODO: Workaround for findOutputValues and buildOutputDefs. See NGCPU-470.
        StringRef funcName;

        // Store the attributes needed by callback
        std::vector<opAttrs> m_attrsVec;
        std::vector<AttrsType> m_attrsTyVec;
    };

    void DialectLoweringPass::runOnOperation()
    {
        // Create type converter and initialize conversion patterns.
        NGraphTypeConverter converter;
        OwningRewritePatternList patterns;

        populateNGraphToAffineConversionPatterns(patterns);

        // Get Memory analysis for in-place memory optimizations
        m_memAnalysis = &getAnalysis<MemoryAnalysis>();

        // Create target that defines legal ops for nGraph dialect to be lowered to.
        ConversionTarget target(getContext());

        target.addLegalDialect<AffineDialect, StandardOpsDialect, LLVM::LLVMDialect>();
        target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
        target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
            // FuncOp is legal only if types have been converted to Std types.
            return typeConverter.isSignatureLegal(op.getType());
        });

        // Gather functions to be processed. Note that new functions will be added to
        // module as part
        // of the function signature conversion so we have to collect the original
        // ones before hand.
        SmallVector<FuncOp, 2> origFuncOps(getOperation().getOps<FuncOp>());

        for (auto origFunc : origFuncOps)
        {
            // TODO: Workaround for findOutputValues and buildOutputDefs. See NGCPU-470.
            funcName = origFunc.getName();

            // Capture output values by looking for the Return and grabbing the values
            // the order of
            // the returned values matches the order of the lowered func signature for
            // results. This
            // is used to find the arg_id that a defined value maps to if it is an
            // output.
            findOutputValues();

            // NOTE: Function signature conversion creates a new FuncOp that is inserted
            // in the
            // module. References the original FuncOp are no longer valid after this
            // point.
            if (failed(applyFullConversion(origFunc, target, std::move(patterns))))
            {
                emitError(mlir::UnknownLoc::get(&getContext()), "Error lowering nGraph dialect\n");
                signalPassFailure();
            }

            // TODO: Encode no alias attribute as part of the function signature
            // conversion or as a
            // separate rewrite pattern. Retrieve new function after signature
            // conversion.
            if (clEnableBarePtrMemRefLowering)
            {
                insertNoAliasArgAttrs();
            }
        }

        insertInitFunc();
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
        FuncOp f = getOperation().lookupSymbol<mlir::FuncOp>(funcName);
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
        FuncOp f = getOperation().lookupSymbol<mlir::FuncOp>(funcName);
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
                // 1. Linear buffer allocation: If the ng value already has a buffer ID
                // assigned,
                //    we re-use that linear buffer SSA value, else generate an AllocOp.
                // 2. View creation: Create a view with the tensor shape and an N-D to 1
                // map over
                //    the linear buffer.
                // If two memrefs are defined via 2 Views over the same buffer, then they
                // share and
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
        // The values would be computed based on the shape of the input to the ng op
        // we are
        // lowering.
        // E.g. If lowering concat, Value for dynamic concat axis will be the sum of
        // input dims.
        // The lowerer will generate code to compute the dims.
        // This is better be done via std.AllocOp but we need to make it hookable to
        // nGraph
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
            // We have a buffer to map to. Create an multi-dim indentity map view with offset over
            // it.
            auto shape = memRefType.getShape();
            auto map = AffineMap::getMultiDimIdentityMap(shape.size(), rewriter.getContext());
            auto et = memRefType.getElementType();
            MemRefType newMemRefType = MemRefType::get(shape, et, map);
            // FixMe : BitWidth may not always be a multiple of 8
            auto byte_width = (et.getIntOrFloatBitWidth() * offset) / 8;
            auto byte_shift = rewriter.create<mlir::ConstantIndexOp>(
                buffer.getDefiningOp()->getLoc(), byte_width);
            auto viewOp = rewriter.create<mlir::ViewOp>(
                buffer.getDefiningOp()->getLoc(), newMemRefType, buffer, byte_shift, ValueRange{});
            return viewOp.getResult();
        }

        // No buffer, create an atomic memref without underlying buffer
        NGRAPH_CHECK(memRefType.hasStaticShape(), "Dynamic shapes are not supported");

        Value alloc = rewriter.create<mlir::AllocOp>(rewriter.getUnknownLoc(), memRefType);
        memRefsToDealloc.push_back(alloc);
        return alloc;
    }

    /// Add llvm.noalias attribute to all the memref function arguments. We know
    /// that this is safe
    /// by nGraph op semantics.
    void DialectLoweringPass::insertNoAliasArgAttrs()
    {
        FuncOp func = getOperation().lookupSymbol<mlir::FuncOp>(funcName);
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
        auto module = getOperation();
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

    mlir::LLVM::GlobalOp DialectLoweringPass::getGlobalOp(StringRef name,
                                                          LLVM::LLVMType globalType,
                                                          bool isConstant,
                                                          LLVM::Linkage linkageType,
                                                          Attribute initVal,
                                                          OpBuilder& rewriter)
    {
        auto module = getOperation();
        auto globalVal = module.lookupSymbol<LLVM::GlobalOp>(name);
        if (!globalVal)
        {
            // Create a global and insert to the module.
            PatternRewriter::InsertionGuard insertGuard(rewriter);
            rewriter.setInsertionPointToStart(module.getBody());
            rewriter.create<LLVM::GlobalOp>(
                rewriter.getUnknownLoc(), globalType, isConstant, linkageType, name, initVal);
        }
        return module.lookupSymbol<LLVM::GlobalOp>(name);
    }

    // Attribute is int64_t
    static void initINT(MLIRContext* context,
                        int64_t intAttr,
                        LLVM::AddressOfOp globalPtr,
                        OpBuilder& builder)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto castOp =
            builder.create<LLVM::BitcastOp>(builder.getUnknownLoc(),
                                            getLLVMType(AttrsType::INT, context).getPointerTo(),
                                            globalPtr);
        auto intOp = builder.create<LLVM::ConstantOp>(
            builder.getUnknownLoc(), llvmI64Ty, builder.getI64IntegerAttr(intAttr));
        builder.create<LLVM::StoreOp>(builder.getUnknownLoc(), intOp, castOp);
    }

    /*
            template <int N>
            struct convAttrs
            {
                bool withRelu;
                int64_t windowStrides[N];
                int64_t windowDilation[N];
                int64_t padBelow[N];
                int64_t padAbove[N];
            };
     */
    static void initCONV1D(MLIRContext* context,
                           convAttrs<1>& convAttrs1d,
                           SmallVector<LLVM::ConstantOp, 12>& constants,
                           LLVM::AddressOfOp globalPtr,
                           OpBuilder& builder)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto llvmI64PtrTy = llvmI64Ty.getPointerTo();
        auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(context);
        auto llvmArray1DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 1);
        auto conv1dTy = getLLVMType(AttrsType::CONV1D, context);
        auto castOp = builder.create<LLVM::BitcastOp>(
            builder.getUnknownLoc(), conv1dTy.getPointerTo(), globalPtr);

        SmallVector<LLVM::GEPOp, 6> geps;
        SmallVector<LLVM::LLVMType, 6> elemsTy{
            llvmI8Ty, llvmArray1DI64Ty, llvmArray1DI64Ty, llvmArray1DI64Ty, llvmArray1DI64Ty};
        for (auto j = 0; j < 5; j++)
        {
            auto gepConv1dOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            elemsTy[j].getPointerTo(),
                                            castOp,
                                            ArrayRef<Value>({constants[0], constants[j]}));
            geps.push_back(gepConv1dOp);
        }
        // Store attribute values
        createStore(llvmI8Ty,
                    builder.getI8IntegerAttr(static_cast<int8_t>(convAttrs1d.withRelu)),
                    geps[0],
                    builder);
        int k = 1;
        for (auto& convAttr : {convAttrs1d.windowStrides[0],
                               convAttrs1d.windowDilation[0],
                               convAttrs1d.padBelow[0],
                               convAttrs1d.padAbove[0]})
        {
            auto gepStructOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            llvmI64PtrTy,
                                            geps[k],
                                            ArrayRef<Value>({constants[0], constants[0]}));
            createStore(llvmI64Ty, builder.getI64IntegerAttr(convAttr), gepStructOp, builder);
            k++;
        }
    }

    static void initCONV2D(MLIRContext* context,
                           convAttrs<2>& convAttrs2d,
                           SmallVector<LLVM::ConstantOp, 12>& constants,
                           LLVM::AddressOfOp globalPtr,
                           OpBuilder& builder)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto llvmI64PtrTy = llvmI64Ty.getPointerTo();
        auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(context);
        auto llvmArray2DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 2);
        auto conv2dTy = getLLVMType(AttrsType::CONV2D, context);
        auto castOp = builder.create<LLVM::BitcastOp>(
            builder.getUnknownLoc(), conv2dTy.getPointerTo(), globalPtr);

        SmallVector<LLVM::GEPOp, 6> geps;
        SmallVector<LLVM::LLVMType, 6> elemsTy{
            llvmI8Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty};
        for (auto j = 0; j < 5; j++)
        {
            auto gepConv2dOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            elemsTy[j].getPointerTo(),
                                            castOp,
                                            ArrayRef<Value>({constants[0], constants[j]}));
            geps.push_back(gepConv2dOp);
        }
        // Store attribute values
        createStore(llvmI8Ty,
                    builder.getI8IntegerAttr(static_cast<int8_t>(convAttrs2d.withRelu)),
                    geps[0],
                    builder);
        int k = 1, m = 0;
        for (auto& convAttr : {convAttrs2d.windowStrides[0],
                               convAttrs2d.windowStrides[1],
                               convAttrs2d.windowDilation[0],
                               convAttrs2d.windowDilation[1],
                               convAttrs2d.padBelow[0],
                               convAttrs2d.padBelow[1],
                               convAttrs2d.padAbove[0],
                               convAttrs2d.padAbove[1]})
        {
            auto gepStructOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            llvmI64PtrTy,
                                            geps[k],
                                            ArrayRef<Value>({constants[0], constants[m]}));
            createStore(llvmI64Ty, builder.getI64IntegerAttr(convAttr), gepStructOp, builder);
            // k increments after every 2 iterations
            if (m == 1)
            {
                k++;
            }
            // m be 0 or 1 alternatively
            m = (m + 1) % 2;
        }
    }

    static void initCONV3D(MLIRContext* context,
                           convAttrs<3>& convAttrs3d,
                           SmallVector<LLVM::ConstantOp, 12>& constants,
                           LLVM::AddressOfOp globalPtr,
                           OpBuilder& builder)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto llvmI64PtrTy = llvmI64Ty.getPointerTo();
        auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(context);
        auto llvmArray3DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 3);
        auto conv3dTy = getLLVMType(AttrsType::CONV3D, context);
        auto castOp = builder.create<LLVM::BitcastOp>(
            builder.getUnknownLoc(), conv3dTy.getPointerTo(), globalPtr);

        SmallVector<LLVM::GEPOp, 6> geps;
        SmallVector<LLVM::LLVMType, 6> elemsTy{
            llvmI8Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty};
        for (auto j = 0; j < 5; j++)
        {
            auto gepConv3dOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            elemsTy[j].getPointerTo(),
                                            castOp,
                                            ArrayRef<Value>({constants[0], constants[j]}));
            geps.push_back(gepConv3dOp);
        }
        // Store attribute values
        createStore(llvmI8Ty,
                    builder.getI8IntegerAttr(static_cast<int8_t>(convAttrs3d.withRelu)),
                    geps[0],
                    builder);
        int k = 1, m = 0;
        for (auto& convAttr : {convAttrs3d.windowStrides[0],
                               convAttrs3d.windowStrides[1],
                               convAttrs3d.windowStrides[2],
                               convAttrs3d.windowDilation[0],
                               convAttrs3d.windowDilation[1],
                               convAttrs3d.windowDilation[2],
                               convAttrs3d.padBelow[0],
                               convAttrs3d.padBelow[1],
                               convAttrs3d.padBelow[2],
                               convAttrs3d.padAbove[0],
                               convAttrs3d.padAbove[1],
                               convAttrs3d.padAbove[2]})
        {
            auto gepStructOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            llvmI64PtrTy,
                                            geps[k],
                                            ArrayRef<Value>({constants[0], constants[m]}));
            createStore(llvmI64Ty, builder.getI64IntegerAttr(convAttr), gepStructOp, builder);
            // k increments after every 3 iterations
            if (m == 2)
            {
                k++;
            }
            // m be 0, 1, or 2 repeatedly.
            m = (m + 1) % 3;
        }
    }

    /*
            template <int N>
            struct poolAttrs
            {
                bool includePaddingInAvgComputation;
                int64_t windowShape[N];
                int64_t windowStrides[N];
                int64_t padBelow[N];
                int64_t padAbove[N];
            };
     */
    static void initPOOL2D(MLIRContext* context,
                           poolAttrs<2>& poolAttrs2d,
                           SmallVector<LLVM::ConstantOp, 12>& constants,
                           LLVM::AddressOfOp globalPtr,
                           OpBuilder& builder)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto llvmI64PtrTy = llvmI64Ty.getPointerTo();
        auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(context);
        auto llvmArray2DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 2);
        auto pool2dTy = getLLVMType(AttrsType::POOL2D, context);
        auto castOp = builder.create<LLVM::BitcastOp>(
            builder.getUnknownLoc(), pool2dTy.getPointerTo(), globalPtr);

        SmallVector<LLVM::GEPOp, 6> geps;
        SmallVector<LLVM::LLVMType, 6> elemsTy{
            llvmI8Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty, llvmArray2DI64Ty};
        for (auto j = 0; j < 5; j++)
        {
            auto gepPool2dOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            elemsTy[j].getPointerTo(),
                                            castOp,
                                            ArrayRef<Value>({constants[0], constants[j]}));
            geps.push_back(gepPool2dOp);
        }
        // Store attribute values
        createStore(llvmI8Ty,
                    builder.getI8IntegerAttr(
                        static_cast<int8_t>(poolAttrs2d.includePaddingInAvgComputation)),
                    geps[0],
                    builder);
        int k = 1, m = 0;
        for (auto& poolAttr : {poolAttrs2d.windowShape[0],
                               poolAttrs2d.windowShape[1],
                               poolAttrs2d.windowStrides[0],
                               poolAttrs2d.windowStrides[1],
                               poolAttrs2d.padBelow[0],
                               poolAttrs2d.padBelow[1],
                               poolAttrs2d.padAbove[0],
                               poolAttrs2d.padAbove[1]})
        {
            auto gepStructOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            llvmI64PtrTy,
                                            geps[k],
                                            ArrayRef<Value>({constants[0], constants[m]}));
            createStore(llvmI64Ty, builder.getI64IntegerAttr(poolAttr), gepStructOp, builder);
            // k increments after every 2 iterations
            if (m == 1)
            {
                k++;
            }
            // m be 0 or 1 alternatively
            m = (m + 1) % 2;
        }
    }

    static void initPOOL3D(MLIRContext* context,
                           poolAttrs<3>& poolAttrs3d,
                           SmallVector<LLVM::ConstantOp, 12>& constants,
                           LLVM::AddressOfOp globalPtr,
                           OpBuilder& builder)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto llvmI64PtrTy = llvmI64Ty.getPointerTo();
        auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(context);
        auto llvmArray3DI64Ty = LLVM::LLVMType::getArrayTy(llvmI64Ty, 3);
        auto pool3dTy = getLLVMType(AttrsType::POOL3D, context);
        auto castOp = builder.create<LLVM::BitcastOp>(
            builder.getUnknownLoc(), pool3dTy.getPointerTo(), globalPtr);

        SmallVector<LLVM::GEPOp, 6> geps;
        SmallVector<LLVM::LLVMType, 6> elemsTy{
            llvmI8Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty, llvmArray3DI64Ty};
        for (auto j = 0; j < 5; j++)
        {
            auto gepPool3dOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            elemsTy[j].getPointerTo(),
                                            castOp,
                                            ArrayRef<Value>({constants[0], constants[j]}));
            geps.push_back(gepPool3dOp);
        }
        // Store attribute values
        createStore(llvmI8Ty,
                    builder.getI8IntegerAttr(
                        static_cast<int8_t>(poolAttrs3d.includePaddingInAvgComputation)),
                    geps[0],
                    builder);
        int k = 1, m = 0;
        for (auto& poolAttr : {poolAttrs3d.windowShape[0],
                               poolAttrs3d.windowShape[1],
                               poolAttrs3d.windowShape[2],
                               poolAttrs3d.windowStrides[0],
                               poolAttrs3d.windowStrides[1],
                               poolAttrs3d.windowStrides[2],
                               poolAttrs3d.padBelow[0],
                               poolAttrs3d.padBelow[1],
                               poolAttrs3d.padBelow[2],
                               poolAttrs3d.padAbove[0],
                               poolAttrs3d.padAbove[1],
                               poolAttrs3d.padAbove[2]})
        {
            auto gepStructOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            llvmI64PtrTy,
                                            geps[k],
                                            ArrayRef<Value>({constants[0], constants[m]}));
            createStore(llvmI64Ty, builder.getI64IntegerAttr(poolAttr), gepStructOp, builder);
            // k increments after every 3 iterations
            if (m == 2)
            {
                k++;
            }
            // m be 0, 1, or 2 repeatedly
            m = (m + 1) % 3;
        }
    }

    /*
            struct gemmAttrs
            {
                bool transposeA;
                bool transposeB;
                int64_t m;
                int64_t n;
                int64_t k;
                int64_t lda;
                int64_t ldb;
                int64_t ldc;
                float alpha;
                float beta;
                BroadcastType broadcastHint;
            };
     */
    static void initGEMM(MLIRContext* context,
                         gemmAttrs gemmAttrs2d,
                         SmallVector<LLVM::ConstantOp, 12>& constants,
                         LLVM::AddressOfOp globalPtr,
                         OpBuilder& builder)
    {
        auto llvmI64Ty = LLVM::LLVMType::getInt64Ty(context);
        auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(context);
        auto llvmI8Ty = LLVM::LLVMType::getInt8Ty(context);
        auto llvmF32Ty = LLVM::LLVMType::getFloatTy(context);
        auto gemmTy = getLLVMType(AttrsType::GEMM, context);
        auto castOp = builder.create<LLVM::BitcastOp>(
            builder.getUnknownLoc(), gemmTy.getPointerTo(), globalPtr);

        SmallVector<LLVM::GEPOp, 12> geps;
        SmallVector<LLVM::LLVMType, 12> elemsTy{llvmI8Ty,
                                                llvmI8Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmI64Ty,
                                                llvmF32Ty,
                                                llvmF32Ty,
                                                llvmI32Ty};
        for (auto j = 0; j < 11; j++)
        {
            auto gepGemmOp =
                builder.create<LLVM::GEPOp>(builder.getUnknownLoc(),
                                            elemsTy[j].getPointerTo(),
                                            castOp,
                                            ArrayRef<Value>({constants[0], constants[j]}));
            geps.push_back(gepGemmOp);
        }
        // Store attribute values
        int k = 0;
        for (auto& gemmAttr : {gemmAttrs2d.transposeA, gemmAttrs2d.transposeB})
        {
            createStore(llvmI8Ty,
                        builder.getI8IntegerAttr(static_cast<int8_t>(gemmAttr)),
                        geps[k],
                        builder);
            k++;
        }
        for (auto& gemmAttr : {gemmAttrs2d.m,
                               gemmAttrs2d.n,
                               gemmAttrs2d.k,
                               gemmAttrs2d.lda,
                               gemmAttrs2d.ldb,
                               gemmAttrs2d.ldc})
        {
            createStore(llvmI64Ty, builder.getI64IntegerAttr(gemmAttr), geps[k], builder);
            k++;
        }
        for (auto& gemmAttr : {gemmAttrs2d.alpha, gemmAttrs2d.beta})
        {
            createStore(llvmF32Ty, builder.getF32FloatAttr(gemmAttr), geps[k], builder);
            k++;
        }
        createStore(llvmI32Ty,
                    builder.getI32IntegerAttr(static_cast<int32_t>(gemmAttrs2d.broadcastHint)),
                    geps[10],
                    builder);
    }

    void DialectLoweringPass::insertInitFunc()
    {
        auto module = getOperation();
        OpBuilder builder(module.getContext());
        OpBuilder::InsertionGuard moduleInsertionGuard(builder);
        builder.setInsertionPointToStart(module.getBody());

        // Insert init function
        auto funcTy = builder.getFunctionType({}, {});
        SmallVector<NamedAttribute, 4> attributes;
        auto funcOp = builder.create<mlir::FuncOp>(
            builder.getUnknownLoc(), "callback_init", funcTy, attributes);
        // Insert entry block
        auto entry = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entry);

        if (m_attrsVec.size() == 0)
        {
            // No callbacks, just return
            builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
            return;
        }

        // Insert operations into entry block
        auto* context = module.getContext();
        auto llvmI32Ty = LLVM::LLVMType::getInt32Ty(context);

        // constants needed by gep
        // LLVM requires that structure indexes be (vectors of) 32-bit integer
        // constants.
        SmallVector<LLVM::ConstantOp, 12> constants;
        auto maxNumElem = 11;
        for (auto i = 0; i < maxNumElem; i++)
        {
            auto constant = builder.create<LLVM::ConstantOp>(
                builder.getUnknownLoc(), llvmI32Ty, builder.getI32IntegerAttr(i));
            constants.push_back(constant);
        }
        auto globalType = getLLVMType(AttrsType::CONV3D, context);
        int32_t i = 0;
        for (auto attrs : m_attrsVec)
        {
            StringRef name = "globalAttrs" + std::to_string(i);
            LLVM::GlobalOp globalVal = getGlobalOp(name,
                                                   globalType,
                                                   false,
                                                   LLVM::Linkage::Internal,
                                                   builder.getZeroAttr(globalType),
                                                   builder);
            auto globalPtr = builder.create<LLVM::AddressOfOp>(builder.getUnknownLoc(), globalVal);
            switch (m_attrsTyVec[i])
            {
            case AttrsType::INT: initINT(context, attrs.intAttr, globalPtr, builder); break;
            case AttrsType::CONV1D:
                initCONV1D(context, attrs.convAttrs1d, constants, globalPtr, builder);
                break;
            case AttrsType::CONV2D:
                initCONV2D(context, attrs.convAttrs2d, constants, globalPtr, builder);
                break;
            case AttrsType::CONV3D:
                initCONV3D(context, attrs.convAttrs3d, constants, globalPtr, builder);
                break;
            case AttrsType::POOL2D:
                initPOOL2D(context, attrs.poolAttrs2d, constants, globalPtr, builder);
                break;
            case AttrsType::POOL3D:
                initPOOL3D(context, attrs.poolAttrs3d, constants, globalPtr, builder);
                break;
            case AttrsType::GEMM:
                initGEMM(context, attrs.gemmAttrs2d, constants, globalPtr, builder);
                break;
            default: break;
            }
            i++;
        }
        builder.create<mlir::ReturnOp>(builder.getUnknownLoc());
    }

    inline int32_t DialectLoweringPass::insertAttrs(opAttrs attrs, AttrsType type)
    {
        m_attrsVec.push_back(attrs);
        m_attrsTyVec.push_back(type);
        return m_attrsVec.size() - 1;
    }

#define REWRITER(OP)                                                                               \
    LogicalResult OP##Conversion::matchAndRewrite(                                                 \
        Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const

    REWRITER(NGAddOp)
    {
        lowerBinaryElementwise<mlir::NGAddOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGSubOp)
    {
        lowerBinaryElementwise<mlir::NGSubOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGMulOp)
    {
        lowerBinaryElementwise<mlir::NGMulOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGDivOp)
    {
        lowerBinaryElementwise<mlir::NGDivOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGGreaterOp)
    {
        lowerBinaryElementwise<mlir::NGGreaterOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGLessOp)
    {
        lowerBinaryElementwise<mlir::NGLessOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGGreaterEqOp)
    {
        lowerBinaryElementwise<mlir::NGGreaterEqOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGLessEqOp)
    {
        lowerBinaryElementwise<mlir::NGLessEqOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGEqOp)
    {
        lowerBinaryElementwise<mlir::NGEqOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGNotEqOp)
    {
        lowerBinaryElementwise<mlir::NGNotEqOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGMaxOp)
    {
        lowerBinaryElementwise<mlir::NGMaxOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGMinOp)
    {
        lowerBinaryElementwise<mlir::NGMinOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGArgMaxRedOp)
    {
        lowerIndexReduction<mlir::NGArgMaxRedOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGArgMinRedOp)
    {
        lowerIndexReduction<mlir::NGArgMinRedOp>(op, operands, rewriter, pass);
        return success();
    }

    bool is_signed(NGTensorType ngTensorType)
    {
        NGRAPH_CHECK(ngTensorType);
        auto ngIntType = ngTensorType.getElementType().dyn_cast<NGIntegerType>();

        return !(ngIntType && ngIntType.isUnsigned());
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
        // Steps
        auto steps = vLHS.getSteps();

        NGRAPH_CHECK(lhs.getType().isa<MemRefType>());
        Type elemTy = lhs.getType().dyn_cast<MemRefType>().getElementType();

        // get the original (nGraph) tensor type
        // this will allow us to check signedness and lower to correct Affine op
        NGRAPH_CHECK(op->getOperands()[0].getType().isa<NGTensorType>());
        auto ngTensorType = op->getOperands()[0].getType().dyn_cast<NGTensorType>();

        affineLoopNestBuilder(lbs, ubs, steps, [&](ValueRange ivRange) {
            auto ivs = llvm::to_vector<4>(ivRange);
            Value val = iLHS(ivs);
            Value zero = createZeroConstant(elemTy);
            iRes(ivs) =
                std_select(is_signed(ngTensorType) ? sgt(val, zero) : ugt(val, zero), val, zero);
        });

        rewriter.replaceOp(op, {result});
        return success();
    }

    // Negative
    REWRITER(NGNegOp)
    {
        lowerUnaryElementwise<mlir::NGNegOp>(op, operands, rewriter, pass);
        return success();
    }

    REWRITER(NGAbsOp)
    {
        lowerUnaryElementwise<mlir::NGAbsOp>(op, operands, rewriter, pass);
        return success();
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
        NGRAPH_CHECK(lhs, "Unexpected null lhs value in DotOp");
        NGRAPH_CHECK(rhs, "Unexpected null rhs value in DotOp");
        NGRAPH_CHECK(result, "Unexpected null result value in DotOp");

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
        // TODO (dcab): We currently generate a super naive loop nest. Improve loop
        // nest layout.

        MemRefBoundsCapture vRes(result), vLhs(lhs), vRhs(rhs);

        NGRAPH_CHECK(vLhs.rank() == 2 && vRhs.rank() == 2 && vRes.rank() == 2,
                     "Dot operation is only supported for 2D tensors");

        // Create induction variables, lower bounds, upper bounds and steps of the
        // loop nest.
        // It's important to note that MemRefBoundsCapture priovides lb/ub/step info
        // is "reverse
        // order", i.e., fastest varying dimension is the last one, slowest varying
        // dimention is the
        // first one.
        Value n, m, k;
        unsigned nDim = vLhs.fastestVarying() - 1;
        unsigned mDim = vRhs.fastestVarying();
        unsigned kDim = vRhs.fastestVarying();
        Value nLb(vLhs.lb(nDim)), mLb(vLhs.lb(mDim)), kLb(vRhs.lb(kDim));
        Value nUb(vLhs.ub(nDim)), mUb(vLhs.ub(mDim)), kUb(vRhs.ub(kDim));
        int64_t nStep = vLhs.step(nDim), mStep = vLhs.step(mDim), kStep = vRhs.step(kDim);

        // Constants and indexed values to be used inside the loop nest.
        AffineIndexedValue iRes(result), iLhs(lhs), iRhs(rhs);
        Value zeroInit(rewriter.create<ConstantOp>(loc, rewriter.getZeroAttr(elemTy)));

        {
            Value n, k;
            affineLoopBuilder(nLb, nUb, nStep, [&](Value n) {
                affineLoopBuilder(kLb, kUb, kStep, [&](Value k) { iRes(n, k) = zeroInit; });
            });
        }
        affineLoopBuilder(nLb, nUb, nStep, [&](Value n) {
            affineLoopBuilder(mLb, mUb, mStep, [&](Value m) {
                affineLoopBuilder(
                    kLb, kUb, kStep, [&](Value k) { iRes(n, k) += iLhs(n, m) * iRhs(m, k); });
            });
        });

        rewriter.replaceOp(op, {result});

        return success();
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

        // For each operand, generate a separate loop to copy into the target slice of
        // "result".
        // We'll keep track of the slice offsets via concatenation_axis_pos.
        auto concatenationAxis = concat.concatenation_axis().getSExtValue();
        Value concatenationAxisPos(std_constant_index(0));

        for (auto& operand : operands)
        {
            NGRAPH_CHECK(operand, "Unexpected null operand in ConcatOp");

            // Assuming rank = r, and the concatenation axis is A where A<r, we'll be
            // creating
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

            llvm::SmallVector<Value, 5> indexVarLbs;
            llvm::SmallVector<Value, 5> indexVarUbs;
            llvm::SmallVector<int64_t, 5> indexVarSteps;
            for (int i = 0; i < rank; i++)
            {
                indexVarLbs.push_back(vOperand.lb(i));
                indexVarUbs.push_back(vOperand.ub(i));
                indexVarSteps.push_back(vOperand.step(i));
            }

            affineLoopNestBuilder(
                indexVarLbs, indexVarUbs, indexVarSteps, [&](ValueRange indexVarsRange) {
                    auto indexVars = llvm::to_vector<5>(indexVarsRange);
                    AffineIndexedValue ivRes(result);
                    AffineIndexedValue ivOperand(operand);

                    // On the LHS of the assignment, adjust the index
                    // for the concatenation axis.
                    llvm::SmallVector<Value, 5> resIndexHandles;
                    for (int i = 0; i < rank; i++)
                    {
                        resIndexHandles.push_back(i == concatenationAxis
                                                      ? indexVars[i] + Value(concatenationAxisPos)
                                                      : indexVars[i]);
                    }

                    ivRes(resIndexHandles) = ivOperand(indexVars);
                });

            // Move up concatenation_axis_pos for the next operand.
            concatenationAxisPos = Value(concatenationAxisPos) + vOperand.ub(concatenationAxis);
        }

        rewriter.replaceOp(op, {result});
        return success();
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
        SmallVector<Value, 4> paramsLbs, paramsUbs;
        SmallVector<int64_t, 4> paramsSteps;
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

        auto indicesLbs = vIndices.getLbs();
        auto indicesUbs = vIndices.getUbs();
        auto indicesSteps = vIndices.getSteps();

        SmallVector<Value, 8> paramsIndices, resIndices;

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
        //                 res[P_0, P_1, .. P_(A-1), I_0, .., I_(M-1), P_(A+1), ...
        //                 P_(N-1)] =
        //                   params[P_0, P_1, .. P_(A-1), indices[I_0, .., I_(M-1)],
        //                          P_(A+1), ... P_(N-1)];

        affineLoopNestBuilder(indicesLbs, indicesUbs, indicesSteps, [&](ValueRange indicesIVRange) {
            auto indicesIVs = llvm::to_vector<4>(indicesIVRange);
            // Load axis value from indices array and cast it to Index Type
            auto axisIdx =
                ValueBuilder<IndexCastOp>(Value(iIndices(indicesIVs)), rewriter.getIndexType());

            affineLoopNestBuilder(paramsLbs, paramsUbs, paramsSteps, [&](ValueRange paramsIVRange) {
                auto paramsIVs = llvm::to_vector<4>(paramsIVRange);
                // construct indices for param
                // [P_0, P_1, .. P_axis-1, Indices[I0, I1, .. I_k-1], P_axis+1,
                // P_axis+2,
                // .. P_n-1]
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
                // [P_0, P_1, .. P_axis-1, I0, I1, .. I_k-1, P_axis+1, P_axis+2,
                // .. P_n-1]
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
        return success();
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
        return success();
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
        Value lb = std_constant_index(0);
        Value ub = std_constant_index(groups);

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

        affineLoopBuilder(lb, ub, 1, [&](Value iv) {
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
        return success();
    }
    REWRITER(NGReturnOp)
    {
        pass.insertDeallocs(rewriter);
        rewriter.replaceOpWithNewOp<ReturnOp>(op);
        return success();
    }

    // Use callback: Pooling, MatMul, Gemm, Softmax, ConvBias
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

    static LLVM::AddressOfOp getGlobalAddr(int32_t index,
                                           PatternRewriter& rewriter,
                                           DialectLoweringPass& pass,
                                           MLIRContext* context)
    {
        auto globalTy = getLLVMType(AttrsType::CONV3D, context);
        StringRef name = "globalAttrs" + std::to_string(index);
        LLVM::GlobalOp globalVal = pass.getGlobalOp(name,
                                                    globalTy,
                                                    false,
                                                    LLVM::Linkage::Internal,
                                                    rewriter.getZeroAttr(globalTy),
                                                    rewriter);
        auto globalPtr = rewriter.create<LLVM::AddressOfOp>(rewriter.getUnknownLoc(), globalVal);
        return globalPtr;
    }

    REWRITER(NGAvgPoolOp)
    {
        lowerPooling<mlir::NGAvgPoolOp>(op, operands, rewriter, pass, context);
        return success();
    }

    REWRITER(NGAvgPoolBackpropOp)
    {
        lowerPooling<mlir::NGAvgPoolBackpropOp>(op, operands, rewriter, pass, context);
        return success();
    }

    REWRITER(NGMaxPoolOp)
    {
        lowerPooling<mlir::NGMaxPoolOp>(op, operands, rewriter, pass, context);
        return success();
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
        NGRAPH_CHECK(src, "Unexpected null src value in MaxPoolBackprop Op");
        NGRAPH_CHECK(delta, "Unexpected null delta value in MaxPoolBackprop Op");
        NGRAPH_CHECK(result, "Unexpected null result value in MaxPoolBackprop Op");

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
                     "DNNL pooling operation is only supported for 3D and 5D tensors");

        opAttrs attrs;
        int32_t index = 0;
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
            index = pass.insertAttrs(attrs, AttrsType::POOL2D);
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
            index = pass.insertAttrs(attrs, AttrsType::POOL3D);
        }
        // Get callback func
        auto unionTy = getLLVMType(AttrsType::CONV3D, context);
        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        FuncOp callBackFunc = pass.getCallDecl(
            "callback_2_inputs",
            {unrankedMemrefTy, unrankedMemrefTy, unrankedMemrefTy, unionTy.getPointerTo(), int64Ty},
            {},
            rewriter);
        // Insert call
        auto globalPtr = getGlobalAddr(index, rewriter, pass, context);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::MAXPOOLBACKPROP), 64);
        SmallVector<mlir::Value, 4> inputs = {src, delta, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 6> args = {
            outputs[0], outputs[1], outputs[2], globalPtr, opTypeArg};
        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);
        return success();
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
        NGRAPH_CHECK(lhs, "Unexpected null lhs value in MatMulOp");
        NGRAPH_CHECK(rhs, "Unexpected null rhs value in MatMulOp");
        NGRAPH_CHECK(result, "Unexpected null result value in MatMulOp");

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
        attrs.gemmAttrs2d.alpha = 1.0;
        attrs.gemmAttrs2d.beta = 0.0;
        auto index = pass.insertAttrs(attrs, AttrsType::GEMM);
        // Get callback func
        auto unionTy = getLLVMType(AttrsType::CONV3D, context);
        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        auto callBackFunc = pass.getCallDecl(
            "callback_2_inputs",
            {unrankedMemrefTy, unrankedMemrefTy, unrankedMemrefTy, unionTy.getPointerTo(), int64Ty},
            {},
            rewriter);
        // Insert call
        auto globalPtr = getGlobalAddr(index, rewriter, pass, context);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::MATMUL), 64);
        SmallVector<mlir::Value, 4> inputs = {lhs, rhs, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 6> args = {
            outputs[0], outputs[1], outputs[2], globalPtr, opTypeArg};
        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);

        return success();
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
        NGRAPH_CHECK(lhs, "Unexpected null lhs value in GemmOp");
        NGRAPH_CHECK(rhs, "Unexpected null rhs value in GemmOp");
        NGRAPH_CHECK(bias, "Unexpected null bias value in GemmOp");
        NGRAPH_CHECK(result, "Unexpected null result value in GemmOp");

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
        auto index = pass.insertAttrs(attrs, AttrsType::GEMM);
        // Get callback func
        auto unionTy = getLLVMType(AttrsType::CONV3D, context);
        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        auto callBackFunc = pass.getCallDecl("callback_3_inputs",
                                             {unrankedMemrefTy,
                                              unrankedMemrefTy,
                                              unrankedMemrefTy,
                                              unrankedMemrefTy,
                                              unionTy.getPointerTo(),
                                              int64Ty},
                                             {},
                                             rewriter);
        // Insert call
        auto globalPtr = getGlobalAddr(index, rewriter, pass, context);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::GEMM), 64);
        SmallVector<mlir::Value, 4> inputs = {lhs, rhs, bias, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 6> args = {
            outputs[0], outputs[1], outputs[2], outputs[3], globalPtr, opTypeArg};
        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);

        return success();
    }

    REWRITER(NGSoftMaxOp)
    {
        auto softmax = cast<NGSoftMaxOp>(op);
        auto loc = softmax.getLoc();

        // Retrieve/generate Values for operands and result.
        ScopedContext scope(rewriter, loc);
        Value lhs = operands[0];
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(lhs, "Unexpected null lhs value in SoftmaxOp");
        NGRAPH_CHECK(result, "Unexpected null result value in SoftmaxOp");

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
                     "DNNL Softmax operation is only supported for 2D and 4D tensors");

        auto axes = softmax.axes().getValue();
        opAttrs attrs;
        attrs.intAttr = axes[0].cast<IntegerAttr>().getInt();
        auto index = pass.insertAttrs(attrs, AttrsType::INT);
        // Get callback func
        auto unionTy = getLLVMType(AttrsType::CONV3D, context);
        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        FuncOp callBackFunc =
            pass.getCallDecl("callback_1_input",
                             {unrankedMemrefTy, unrankedMemrefTy, unionTy.getPointerTo(), int64Ty},
                             {},
                             rewriter);
        // Insert call
        auto globalPtr = getGlobalAddr(index, rewriter, pass, context);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::SOFTMAX), 64);
        SmallVector<mlir::Value, 4> inputs = {lhs, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 4> args = {outputs[0], outputs[1], globalPtr, opTypeArg};
        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);

        return success();
    }

    REWRITER(NGConvBiasOp)
    {
        auto convBias = cast<NGConvBiasOp>(op);
        auto loc = convBias.getLoc();
        ScopedContext scope(rewriter, loc);

        // Get operands
        Value result = pass.buildOutputDefs(op, rewriter)[0];
        NGRAPH_CHECK(result, "Unexpected null result in ConvBias Op");
        Value images = operands[0];
        Value filters = operands[1];
        Value bias = operands[2];
        auto strides = convBias.strides().getValue();
        auto dilation = convBias.dilation().getValue();
        auto padBelow = convBias.padBelow().getValue();
        auto padAbove = convBias.padBelow().getValue();

        auto resultTy = result.getType().dyn_cast<MemRefType>();
        auto resultShape = resultTy.getShape();
        auto imagesTy = images.getType().dyn_cast<MemRefType>();
        auto imagesShape = imagesTy.getShape();
        NGRAPH_CHECK(resultTy, "Unexpected non-memref result type");
        NGRAPH_CHECK(imagesTy, "Unexpected non-memref LHS type");

        Type elemTy = resultTy.getElementType();
        NGRAPH_CHECK(elemTy == imagesTy.getElementType(), "Types mismatch in ConvBias");

        NGRAPH_CHECK((imagesShape.size() == 3 && resultShape.size() == 3) ||
                         (imagesShape.size() == 4 && resultShape.size() == 4) ||
                         (imagesShape.size() == 5 && resultShape.size() == 5),
                     "DNNL conv operation is only supported for 3D, 4D, and 5D tensors");

        opAttrs attrs;
        size_t index = 0;
        if (imagesShape.size() == 3)
        {
            attrs.convAttrs1d.withRelu = convBias.withRelu();
            attrs.convAttrs1d.windowStrides[0] = strides[0].cast<IntegerAttr>().getInt();
            attrs.convAttrs1d.windowDilation[0] = dilation[0].cast<IntegerAttr>().getInt();
            attrs.convAttrs1d.padBelow[0] = padBelow[0].cast<IntegerAttr>().getInt();
            attrs.convAttrs1d.padAbove[0] = padAbove[0].cast<IntegerAttr>().getInt();
            index = pass.insertAttrs(attrs, AttrsType::CONV1D);
        }
        else if (imagesShape.size() == 4)
        {
            attrs.convAttrs2d.withRelu = convBias.withRelu();
            for (auto i = 0; i < 2; i++)
            {
                attrs.convAttrs2d.windowStrides[i] = strides[i].cast<IntegerAttr>().getInt();
                attrs.convAttrs2d.windowDilation[i] = dilation[i].cast<IntegerAttr>().getInt();
                attrs.convAttrs2d.padBelow[i] = padBelow[i].cast<IntegerAttr>().getInt();
                attrs.convAttrs2d.padAbove[i] = padAbove[i].cast<IntegerAttr>().getInt();
            }
            index = pass.insertAttrs(attrs, AttrsType::CONV2D);
        }
        else if (imagesShape.size() == 5)
        {
            attrs.convAttrs3d.withRelu = convBias.withRelu();
            for (auto i = 0; i < 3; i++)
            {
                attrs.convAttrs3d.windowStrides[i] = strides[i].cast<IntegerAttr>().getInt();
                attrs.convAttrs3d.windowDilation[i] = dilation[i].cast<IntegerAttr>().getInt();
                attrs.convAttrs3d.padBelow[i] = padBelow[i].cast<IntegerAttr>().getInt();
                attrs.convAttrs3d.padAbove[i] = padAbove[i].cast<IntegerAttr>().getInt();
            }
            index = pass.insertAttrs(attrs, AttrsType::CONV3D);
        }
        // Get callback func
        auto unionTy = getLLVMType(AttrsType::CONV3D, context);
        auto int64Ty = rewriter.getIntegerType(64);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        FuncOp callBackFunc = pass.getCallDecl("callback_3_inputs",
                                               {unrankedMemrefTy,
                                                unrankedMemrefTy,
                                                unrankedMemrefTy,
                                                unrankedMemrefTy,
                                                unionTy.getPointerTo(),
                                                int64Ty},
                                               {},
                                               rewriter);
        // Insert call
        auto globalPtr = getGlobalAddr(index, rewriter, pass, context);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(OpType::CONVOLUTIONBIAS), 64);
        SmallVector<mlir::Value, 4> inputs = {images, filters, bias, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 6> args = {
            outputs[0], outputs[1], outputs[2], outputs[3], globalPtr, opTypeArg};
        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);

        return success();
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
        //             Images[n, c, i_1 + j_1, .. i_f + j_f] * Filters[k, c, j_1, ..
        //             j_f]

        // With padding, we check (using IntegerSets) whether each spatial dim in
        // Images lie inside
        // non-padded spatial region. If true, we perform the computation:
        //
        //         for <j_1 .. j_f> : <0 .. 0> -> <F_1 .. F_f>
        //         if(indices in non-padded region):
        //           Output[n, k, r_1, .. r_f] +=
        //             Images[n, c, i_1 + j_1, .. i_f + j_f] * Filters[k, c, j_1, ..
        //             j_f]

        ScopedContext scope(rewriter, loc);
        auto strides = stridesAttr.getValue();
        auto padBelow = padBelowAttr.getValue();
        auto padAbove = padBelowAttr.getValue();
        Type elemTy = images.getType().cast<MemRefType>().getElementType();

        // Create views
        MemRefBoundsCapture vRes(result), vImages(images), vFilters(filters);
        // Create indexed Values
        AffineIndexedValue iRes(result), iImages(images), iFilters(filters);
        // Bounds on batch size N
        Value batchLb = vImages.lb(0), batchUb = vImages.ub(0);
        // Bounds on spatial dimensions
        SmallVector<Value, 4> resSpatialLbs, resSpatialUbs;
        SmallVector<Value, 4> imgSpatialLbs, imgSpatialUbs;
        SmallVector<Value, 4> filtersSpatialLbs, filtersSpatialUbs;
        // Spatial rank
        unsigned spatialRank = vImages.rank() - 2;

        // Result spatial indices and bounds
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
        Value numFiltersLb;
        Value numFiltersUb;
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
                numFiltersLb = Value(kLb);
                numFiltersUb = Value(kUb);
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
        Value numChannelsLb = (cLb == nullptr) ? vImages.lb(1) : Value(cLb);
        Value numChannelsUb = (cUb == nullptr) ? vImages.ub(1) : Value(cUb);

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
        SmallVector<Value, 4> filtersSpatialIndices(spatialRank);

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
            // IntegerSet (d0, d1, .. d_N-1)[LB_0, LB_1, .. LB_N-1, UB_0, UB_1, ..
            // UB_N-1], where
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
            Value c;

            affineLoopBuilder(batchLb, batchUb, 1, [&](Value n) {
                affineLoopBuilder(numFiltersLb, numFiltersUb, 1, [&](Value k) {
                    affineLoopNestBuilder(
                        resSpatialLbs,
                        resSpatialUbs,
                        resSteps,
                        [&](ValueRange resSpatialIndicesRange) {
                            auto resSpatialIndices = llvm::to_vector<4>(resSpatialIndicesRange);
                            SmallVector<Value, 4> resIndices;
                            // Result indices
                            resIndices.push_back(n);
                            if (groupConvolution && groupsInFilters)
                            {
                                // compute global C_OUT from gID and k
                                // gId * C_OUT (num of filters) + k
                                resIndices.push_back(Value(gId) * numFiltersUb + k);
                            }
                            else
                            {
                                resIndices.push_back(k);
                            }
                            resIndices.insert(resIndices.end(),
                                              resSpatialIndices.begin(),
                                              resSpatialIndices.end());
                            Value zero = createZeroConstant(elemTy);
                            iRes(resIndices) = zero;
                        });
                });
            });
        }

        // Convolution loop
        affineLoopBuilder(batchLb, batchUb, 1, [&](Value n) {
            // Number of filters loop
            affineLoopBuilder(numFiltersLb, numFiltersUb, 1, [&](Value k) {
                // Channels loop
                affineLoopBuilder(numChannelsLb, numChannelsUb, 1, [&](Value c) {
                    // Results loop
                    affineLoopNestBuilder(
                        resSpatialLbs,
                        resSpatialUbs,
                        resSteps,
                        [&](ValueRange resSpatialIndicesRange) {
                            auto resSpatialIndices = llvm::to_vector<4>(resSpatialIndicesRange);
                            // Compute image start indices
                            SmallVector<Value, 4> imgStartIndices;
                            for (auto i = 0; i < spatialRank; i++)
                            {
                                IntegerAttr iAttr = strides[i].cast<IntegerAttr>();
                                auto stride = std_constant_index(iAttr.getInt());
                                imgStartIndices.push_back(resSpatialIndices[i] * stride);
                            }
                            SmallVector<Value, 4> resIndices;
                            // Result indices
                            resIndices.push_back(n);
                            if (groupConvolution && groupsInFilters)
                            {
                                // gId * C_OUT (num of filters) + k
                                resIndices.push_back(Value(gId) * numFiltersUb + k);
                            }
                            else
                            {
                                resIndices.push_back(k);
                            }

                            resIndices.insert(resIndices.end(),
                                              resSpatialIndices.begin(),
                                              resSpatialIndices.end());
                            // Filters spatial loop
                            affineLoopNestBuilder(
                                filtersSpatialLbs,
                                filtersSpatialUbs,
                                filtersSteps,
                                [&](ValueRange filtersSpatialIndicesRange) {
                                    auto filtersSpatialIndices =
                                        llvm::to_vector<4>(filtersSpatialIndicesRange);
                                    SmallVector<Value, 4> imgIndices, filtersIndices;
                                    // Image indices
                                    // Here we compute the virtual start index into the padded
                                    // image.
                                    imgIndices.push_back(n);
                                    imgIndices.push_back(c);
                                    for (auto i = 0; i < spatialRank; i++)
                                    {
                                        imgIndices.push_back(imgStartIndices[i] +
                                                             filtersSpatialIndices[i]);
                                    }
                                    // Filter indices

                                    // If we are doing group convolution and filters shape dim0
                                    // holds the number of groups, we need to use group id as
                                    // the first index
                                    if (groupConvolution && groupsInFilters)
                                    {
                                        filtersIndices.push_back(Value(gId));
                                    }

                                    filtersIndices.push_back(k);
                                    // subtract lower bound of channel
                                    // if we are doing group convolution this bound will advance
                                    // based on the group id. For the filters, it should always
                                    // start from 0
                                    filtersIndices.push_back(c - numChannelsLb);
                                    filtersIndices.insert(filtersIndices.end(),
                                                          filtersSpatialIndices.begin(),
                                                          filtersSpatialIndices.end());

                                    if (withPadding)
                                    {
                                        // if args : img dims, img lbs, img ubs
                                        SmallVector<Value, 4>::iterator it = imgIndices.begin();
                                        std::advance(it, 2);
                                        SmallVector<Value, 4> affineIfArgs(it, imgIndices.end());
                                        affineIfArgs.insert(affineIfArgs.end(),
                                                            imgSpatialLbs.begin(),
                                                            imgSpatialLbs.end());
                                        affineIfArgs.insert(affineIfArgs.end(),
                                                            imgSpatialUbs.begin(),
                                                            imgSpatialUbs.end());

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
                                            SmallVector<Value, 4> adjustedImgIndices;
                                            adjustedImgIndices.push_back(n);
                                            adjustedImgIndices.push_back(c);
                                            for (auto i = 0; i < spatialRank; i++)
                                            {
                                                adjustedImgIndices.push_back(
                                                    imgIndices[2 + i] -
                                                    std_constant_index(padBelowIntValues[i]));
                                            }
                                            iRes(resIndices) =
                                                iRes(resIndices) + (iImages(adjustedImgIndices) *
                                                                    iFilters(filtersIndices));
                                        }
                                    }
                                    else
                                    {
                                        iRes(resIndices) =
                                            iRes(resIndices) +
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
        // Steps
        auto steps = vLHS.getSteps();

        NGRAPH_CHECK(lhs.getType().isa<MemRefType>());
        Type elemTy = lhs.getType().cast<MemRefType>().getElementType();
        auto ngTensorType = op->getOperands()[0].getType().dyn_cast<NGTensorType>();

        affineLoopNestBuilder(lbs, ubs, steps, [&](ValueRange ivRange) {
            auto ivs = llvm::to_vector<4>(ivRange);
            Value val = iLHS(ivs);
            if (isa<NGNegOp>(op))
            {
                Value zero = createZeroConstant(elemTy);
                iRes(ivs) = zero - val;
            }
            else if (isa<NGAbsOp>(op))
            {
                Value zero = createZeroConstant(elemTy);
                iRes(ivs) = std_select(
                    is_signed(ngTensorType) ? slt(val, zero) : ult(val, zero), zero - val, val);
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
        // Steps
        auto steps = vLHS.getSteps();
        // element type of the operand
        Type elemTy = result.getType().cast<MemRefType>().getElementType();

        // get the original (nGraph) tensor type
        // this will allow us to check signedness and lower to correct Affine op
        NGRAPH_CHECK(op->getOperands()[0].getType().isa<NGTensorType>());
        auto ngTensorType = op->getOperands()[0].getType().dyn_cast<NGTensorType>();

        affineLoopNestBuilder(
            lbs,
            ubs,
            steps,
            // single stmt body
            [&](ValueRange ivRange) {
                auto ivs = llvm::to_vector<4>(ivRange);
                auto left = Value(iLHS(ivs));
                auto right = Value(iRHS(ivs));

                if (isa<NGAddOp>(op))
                {
                    iRes(ivs) = left + right;
                }
                else if (isa<NGSubOp>(op))
                {
                    iRes(ivs) = left - right;
                }
                else if (isa<NGMulOp>(op))
                {
                    iRes(ivs) = left * right;
                }
                else if (isa<NGDivOp>(op))
                {
                    iRes(ivs) = left / right;
                }
                // TODO(pthoreho) For all comparision operators, use
                // zero_extendi(Value(iLHS(ivs)) !=
                // Value(iRHS(ivs)), IntegerType::get(8, op->getContext()));
                // instead of std_select once `zero_extendi` is
                // made available in the edsc::intrinsics namescope in MLIR repo.
                else if (isa<NGGreaterOp>(op))
                {
                    auto ones = createOneConstant(elemTy);
                    auto zeros = createZeroConstant(elemTy);
                    iRes(ivs) = std_select(
                        is_signed(ngTensorType) ? sgt(left, right) : ugt(left, right), ones, zeros);
                }
                else if (isa<NGLessOp>(op))
                {
                    auto ones = createOneConstant(elemTy);
                    auto zeros = createZeroConstant(elemTy);
                    iRes(ivs) = std_select(
                        is_signed(ngTensorType) ? slt(left, right) : ult(left, right), ones, zeros);
                }
                else if (isa<NGGreaterEqOp>(op))
                {
                    auto ones = createOneConstant(elemTy);
                    auto zeros = createZeroConstant(elemTy);
                    iRes(ivs) = std_select(
                        is_signed(ngTensorType) ? sge(left, right) : uge(left, right), ones, zeros);
                }
                else if (isa<NGLessEqOp>(op))
                {
                    auto ones = createOneConstant(elemTy);
                    auto zeros = createZeroConstant(elemTy);
                    iRes(ivs) = std_select(
                        is_signed(ngTensorType) ? sle(left, right) : ule(left, right), ones, zeros);
                }
                else if (isa<NGEqOp>(op))
                {
                    auto ones = createOneConstant(elemTy);
                    auto zeros = createZeroConstant(elemTy);
                    iRes(ivs) = std_select(eq(left, right), ones, zeros);
                }
                else if (isa<NGNotEqOp>(op))
                {
                    auto ones = createOneConstant(elemTy);
                    auto zeros = createZeroConstant(elemTy);
                    iRes(ivs) = std_select(ne(left, right), ones, zeros);
                }
                else if (isa<NGMaxOp>(op))
                {
                    iRes(ivs) = std_select(
                        is_signed(ngTensorType) ? sgt(left, right) : ugt(left, right), left, right);
                }
                else if (isa<NGMinOp>(op))
                {
                    iRes(ivs) = std_select(
                        is_signed(ngTensorType) ? slt(left, right) : ult(left, right), left, right);
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

        // get the original (nGraph) tensor type
        // this will allow us to check signedness and lower to correct Affine op
        NGRAPH_CHECK(op->getOperands()[0].getType().isa<NGTensorType>());
        auto ngTensorType = op->getOperands()[0].getType().dyn_cast<NGTensorType>();

        // Generate loop nest that initializes result to lower bound of the axis to be
        // reduced.
        {
            auto steps = vRes.getSteps();
            auto initVal = vArg.lb(axis);
            affineLoopNestBuilder(resLbs, resUbs, steps, [&](ValueRange ivRange) {
                auto ivs = llvm::to_vector<4>(ivRange);
                iRes(ivs) = ValueBuilder<IndexCastOp>(initVal, resTy);
            });
        }

        // Generate loop nest that computes the actual index reduction.
        {
            auto steps = vArg.getSteps();
            SmallVector<Value, 8> nonRedIVs;
            SmallVector<Value, 8> tempIVs;

            Type resTy = result.getType().cast<MemRefType>().getElementType();
            NGRAPH_CHECK(resTy.isa<IntegerType>(),
                         "Expected integer result type in index reduction");

            // iterate over all argument dimensions
            affineLoopNestBuilder(argLbs, argUbs, steps, [&](ValueRange allIVRange) {
                auto allIVs = llvm::to_vector<4>(allIVRange);
                // build a list of non-reduction IVs
                for (auto i = 0; i < vArg.rank(); i++)
                {
                    if (i != axis)
                    {
                        nonRedIVs.push_back(allIVs[i]);
                    }
                }

                // Load current min index with integer data type and convert it to
                // index data type.
                auto currRedIdx = ValueBuilder<IndexCastOp>(Value(iRes(nonRedIVs)),
                                                            IndexType::get(resTy.getContext()));

                // Build list of IVs including current min index.
                for (auto i = 0; i < vArg.rank(); i++)
                {
                    if (i != axis)
                    {
                        tempIVs.push_back(allIVs[i]);
                    }
                    else
                    {
                        tempIVs.push_back(currRedIdx);
                    }
                }
                Value newRedIdx = std::is_same<RedOp, NGArgMinRedOp>()
                                      ? std_select(is_signed(ngTensorType)
                                                       ? slt(affineArg(allIVs), stdArg(tempIVs))
                                                       : ult(affineArg(allIVs), stdArg(tempIVs)),
                                                   allIVs[axis],
                                                   currRedIdx)
                                      : std_select(is_signed(ngTensorType)
                                                       ? slt(stdArg(tempIVs), affineArg(allIVs))
                                                       : ult(stdArg(tempIVs), affineArg(allIVs)),
                                                   allIVs[axis],
                                                   currRedIdx);

                iRes(nonRedIVs) = ValueBuilder<IndexCastOp>(newRedIdx, resTy);
            });
        }

        rewriter.replaceOp(op, result);
    }

    template <typename OP>
    void lowerPooling(Operation* op,
                      ArrayRef<Value> operands,
                      PatternRewriter& rewriter,
                      DialectLoweringPass& pass,
                      MLIRContext* context)
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
        NGRAPH_CHECK(lhs, "Unexpected null lhs value in Pooling Op");
        NGRAPH_CHECK(result, "Unexpected null result value in Pooling Op");

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
                     "DNNL pooling operation is only supported for 3D and 5D tensors");

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

        opAttrs attrs;
        size_t index = 0;
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
            index = pass.insertAttrs(attrs, AttrsType::POOL2D);
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
            index = pass.insertAttrs(attrs, AttrsType::POOL3D);
        }
        // Get callback func
        auto unionTy = getLLVMType(AttrsType::CONV3D, context);
        auto unrankedMemrefTy = UnrankedMemRefType::get(elemTy, 0);
        FuncOp callBackFunc =
            pass.getCallDecl("callback_1_input",
                             {unrankedMemrefTy, unrankedMemrefTy, unionTy.getPointerTo(), int64Ty},
                             {},
                             rewriter);
        // Insert call
        auto globalPtr = getGlobalAddr(index, rewriter, pass, context);
        auto opTypeArg = rewriter.create<mlir::ConstantIntOp>(
            rewriter.getUnknownLoc(), static_cast<int64_t>(ty), 64);
        SmallVector<mlir::Value, 4> inputs = {lhs, result};
        SmallVector<mlir::Value, 4> outputs;
        castMemRef(inputs, outputs, rewriter, unrankedMemrefTy);
        SmallVector<mlir::Value, 4> args = {outputs[0], outputs[1], globalPtr, opTypeArg};
        rewriter.create<mlir::CallOp>(rewriter.getUnknownLoc(), callBackFunc, args);
        rewriter.replaceOp(op, result);
    }

    Value createZeroConstant(mlir::Type type)
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

    Value createOneConstant(mlir::Type type)
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
}

namespace mlir
{
    std::unique_ptr<Pass> createDialectLoweringPass()
    {
        return std::make_unique<DialectLoweringPass>();
    }
}

static PassRegistration<DialectLoweringPass> pass(PASS_NAME,
                                                  "Convert nGraph dialect to affine dialect");
