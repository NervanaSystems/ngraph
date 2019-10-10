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

// NOTE: This file follows nGraph format style.
// Follows nGraph naming convention for public APIs only, else MLIR naming convention.

#include "compiler.hpp"

#include "dialect/dialect.hpp"
#include "dialect/ops.hpp"
#include "dialect/type.hpp"
#include "lowerer.hpp"
#include "ngraph/check.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "ngraph/type/element_type.hpp"
#include "pass/memory_optimization.hpp"
#include "tools.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <mutex>

// Defines a new LLVM debug type for this file to be used by LLVM_DEBUG macro.
#define DEBUG_TYPE "mlir-compiler"

using llvm::SmallVector;
using llvm::StringRef;
using llvm::ArrayRef;

using namespace ngraph;
using namespace ngraph::runtime::ngmlir;

// *** Debug flags ***

static llvm::cl::opt<bool> clPrintIRAfterAll(
    "ngraph-print-ir-after-all",
    llvm::cl::init(false),
    llvm::cl::desc(
        "Print IR after transformation that are not implemented as passes in the MLIRCompiler. It "
        "complements MLIR -print-ir-after-all and LLVM -print-after-all flags"));

// *** Optimization flags ***

static llvm::cl::opt<bool> clEnableNgInPlaceMemoryOpt(
    "ng-inplace-mem-opt",
    llvm::cl::init(false),
    llvm::cl::desc("Enable ngraph dialect in-place memory optimization pass"));

static llvm::cl::opt<bool>
    clEnableAffineLoopFusion("ngraph-affine-loop-fusion",
                             llvm::cl::init(false),
                             llvm::cl::desc("Enable loop fusion optimization in Affine dialect"));

static llvm::cl::opt<bool>
    clEnableAffineLoopTiling("ngraph-affine-loop-tile",
                             llvm::cl::init(false),
                             llvm::cl::desc("Enable loop tiling optimization in Affine dialect"));

static llvm::cl::opt<unsigned>
    clLoopTilingCacheLevel("ngraph-affine-loop-tile-cache-level",
                           llvm::cl::init(2),
                           llvm::cl::desc("Cache level to which to apply affine loop tiling."));

static llvm::cl::opt<unsigned> clLoopTilingCacheSize(
    "ngraph-affine-loop-tile-cache-size",
    llvm::cl::init(0),
    llvm::cl::desc(
        "Cache size to use in affine loop tiling. If not zero, it overrides the cache-size "
        "inferred from the host CPU using for the cache level specified by "
        "-ngraph-loop-tile-cache-level."));

// *** Debug flags ***

static llvm::cl::opt<bool>
    clDumpObjectFile("ngraph-dump-mlir-object-file",
                     llvm::cl::desc("Dump MLIR JITted-compiled object to file specified with "
                                    "-object-filename (<input file>.o by default)."));

static llvm::cl::opt<std::string>
    clObjectFilename("ngraph-mlir-object-filename",
                     llvm::cl::desc("Dump MLIR JITted-compiled object to file jitted_mlir.o"));

#define COMPILE_OP_DECL(op_name)                                                                   \
    createOp<op_name>(MLIRCompiler & compiler, const ngraph::Node* ngNode)

// Default optimization level.
llvm::CodeGenOpt::Level MLIRCompiler::mlirOptLevel = llvm::CodeGenOpt::Level::Aggressive;

// Target machine will be properly initialized by `init_mlir`.
std::unique_ptr<llvm::TargetMachine> MLIRCompiler::targetMachine;

/// Creates target machine for current host.
static llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
    createDefaultTargetMachine(unsigned optLevel)
{
    auto machineBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!machineBuilder)
    {
        return machineBuilder.takeError();
    }

    // Relocation model and code model are kept to default values. CodeGen optimization level
    // matches LLVM recommendations, i.e.:
    // enum Level {
    //   None,        // -O0
    //   Less,        // -O1
    //   Default,     // -O2, -Os
    //   Aggressive   // -O3
    // };
    machineBuilder->setCodeGenOptLevel((llvm::CodeGenOpt::Level)optLevel);
    return machineBuilder->createTargetMachine();
}

void MLIRCompiler::init_mlir()
{
    // Mutex to safely initialize MLIR.
    static std::mutex mlirInitMutex;
    static bool initialized = false;

    std::unique_lock<std::mutex> lock(mlirInitMutex);

    if (!initialized)
    {
        initializeNGraphMLIR();

        // Register MLIR command line options in the pool of supported flags and and process flags
        // from environment variable to be used by nGraph, MLIR and LLVM.
        mlir::registerPassManagerCLOptions();
        llvm::cl::ParseEnvironmentOptions("ngraph", "NGRAPH_MLIR_OPTIONS", "");

        // Override default optimization level with macro value.
        if (char* optLevelStr = std::getenv("NGRAPH_MLIR_OPT_LEVEL"))
        {
            unsigned clOptLevel = std::stoi(optLevelStr);
            NGRAPH_CHECK(clOptLevel >= 0 && clOptLevel <= 3, "Invalid optimization level");
            mlirOptLevel = (llvm::CodeGenOpt::Level)clOptLevel;
        }

        // Initialize LLVM targets and target machine for current host.
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        auto expectedTargetMachine = createDefaultTargetMachine(mlirOptLevel);
        NGRAPH_CHECK(expectedTargetMachine, "Invalid target machine");
        targetMachine = std::move(*expectedTargetMachine);

        initialized = true;
    }
}

void MLIRCompiler::compile()
{
    buildNgDialectModule();
    optimizeNgDialect();
    lowerNgDialect();
}

void MLIRCompiler::run(std::vector<void*>& externalTensors)
{
    bindArguments(externalTensors);
    execute();
    cleanup();
}

// Creates an MLIR module and function with nGraph dialect ops from the input CompiledKernel.
void MLIRCompiler::buildNgDialectModule()
{
    // initialize an empty module
    m_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&m_context));

    TypeList argsTypeList, resultTypeList;

    // Retrieve input and output tensors.
    const auto& kernelInputs = m_compiledKernel->get_arguments();
    const auto& kernelOutput = m_compiledKernel->get_kernel_outputs();
    NGRAPH_CHECK(kernelInputs.size() != 0, "Cannot have empty inputs list");
    NGRAPH_CHECK(kernelOutput.size() != 0, "Cannot have empty outputs list");

    for (auto input : kernelInputs)
    {
        argsTypeList.push_back(getMlirType(input.get()));
    }

    for (auto output : kernelOutput)
    {
        resultTypeList.push_back(getMlirType(output.get()));
    }

    auto funcType = mlir::FunctionType::get(argsTypeList, resultTypeList, &m_context);
    auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(&m_context), "main", funcType);
    function.addEntryBlock();

    // populate Tensor->Value maps
    int i = 0;
    for (auto input : kernelInputs)
    {
        mlir::Value* arg = function.getArgument(i);
        TensorInfo tensorInfo{arg};
        m_tensorToValueMap.insert(TensorToInfo(input->get_output_tensor_ptr().get(), tensorInfo));
        i++;
    }

    // create builder
    m_builder = std::unique_ptr<mlir::OpBuilder>(new mlir::OpBuilder(function.getBody()));
    buildNgDialect();
    m_module->push_back(function);
    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Invalid module after lowering to NG dialect");
    }

    dumpMlirModule("nGraph Dialect Construction");
}

template <typename T>
void MLIRCompiler::getMlirShape(T ngShape, llvm::SmallVectorImpl<int64_t>& mlirShape)
{
    for (auto dim : ngShape)
    {
        mlirShape.push_back(dim);
    }
}

template <typename T>
mlir::ArrayAttr MLIRCompiler::getShapeAsAttr(T ngShape)
{
    SmallVector<int64_t, 4> mlirShape;
    getMlirShape(ngShape, mlirShape);
    return m_builder->getI64ArrayAttr(mlirShape);
}

// Converts an nGraph Tensor into an MLIR tensor type, including the conversion of the Tensor's
// element type.
mlir::Type MLIRCompiler::getMlirType(const descriptor::Tensor* tensor)
{
    llvm::SmallVector<int64_t, 4> mlirShape;
    getMlirShape(tensor->get_shape(), mlirShape);
    return mlir::NGTensorType::get(&m_context, getMlirType(tensor->get_element_type()), mlirShape);
}

// Converts an nGraph element type into an MLIR type.
mlir::Type MLIRCompiler::getMlirType(const element::Type& type)
{
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

    switch (type)
    {
    case ngraph::element::Type_t::undefined:
    case ngraph::element::Type_t::dynamic:
    default: NGRAPH_CHECK(false, "MLIR: Unsupported NGraph types"); break;
    case ngraph::element::Type_t::bf16: return mlir::NGFloatType::getBF16(&m_context);
    case ngraph::element::Type_t::f16: return mlir::NGFloatType::getF16(&m_context);
    case ngraph::element::Type_t::f32: return mlir::NGFloatType::getF32(&m_context);
    case ngraph::element::Type_t::f64: return mlir::NGFloatType::getF64(&m_context);
    case ngraph::element::Type_t::i8: return mlir::NGIntegerType::getInt8(&m_context);
    case ngraph::element::Type_t::u8:
    case ngraph::element::Type_t::boolean: return mlir::NGIntegerType::getUInt8(&m_context);
    case ngraph::element::Type_t::i16: return mlir::NGIntegerType::getInt16(&m_context);
    case ngraph::element::Type_t::u16: return mlir::NGIntegerType::getInt16(&m_context);
    case ngraph::element::Type_t::i32: return mlir::NGIntegerType::getInt32(&m_context);
    case ngraph::element::Type_t::u32: return mlir::NGIntegerType::getUInt32(&m_context);
    case ngraph::element::Type_t::i64: return mlir::NGIntegerType::getInt64(&m_context);
    case ngraph::element::Type_t::u64: return mlir::NGIntegerType::getUInt64(&m_context);
    }
    NGRAPH_CHECK(false, "Unreachable");
    return mlir::Type();

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif
}

mlir::Type MLIRCompiler::getMlirType(const ngraph::Node* node)
{
    descriptor::Tensor* outTensor = node->get_output_tensor_ptr().get();
    return getMlirType(outTensor);
}

void MLIRCompiler::updateTensorValue(descriptor::Tensor* tensor, mlir::Value* value)
{
    NGRAPH_CHECK(m_tensorToValueMap.find(tensor) == m_tensorToValueMap.end(),
                 "tensor value already defined");
    TensorInfo tensorInfo{value};
    m_tensorToValueMap.insert(TensorToInfo(tensor, tensorInfo));
}

MLIRCompiler::TensorInfo MLIRCompiler::getTensorValue(descriptor::Tensor* tensor)
{
    auto it = m_tensorToValueMap.find(tensor);

    NGRAPH_CHECK(it != m_tensorToValueMap.end(), "Undefined tensor");

    return it->second;
}

// Lowers nGraph dialect all the way to LLVM module.
void MLIRCompiler::lowerNgDialect()
{
    // Lower NG dialect to Affine
    mlir::PassManager pm(&m_context);
    pm.addPass(mlir::createDialectLoweringPass());
    pm.addPass(mlir::createCanonicalizerPass());

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Incorrect module after dialect lowering");
    }

    optimize();

    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    // Lower Standard dialect to LLVM dialect.
    mlir::LLVMTypeConverter llvmConverter(&m_context);
    mlir::OwningRewritePatternList patterns;
    mlir::populateLoopToStdConversionPatterns(patterns, &m_context);
    mlir::populateStdToLLVMConversionPatterns(llvmConverter, patterns);

    mlir::ConversionTarget target(m_context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>(
        [&](mlir::FuncOp op) { return llvmConverter.isSignatureLegal(op.getType()); });
    auto result = applyFullConversion(*m_module, target, std::move(patterns), &llvmConverter);
    NGRAPH_CHECK(succeeded(result), "Standard to LLVM dialect conversion failed");

    dumpMlirModule("LLVM-IR Dialect Conversion");

    // Create an MLIR execution engine. We use a null MLIR pass manager for now to make sure we
    // don't run MLIR passes that were already run. We also pass a default transformer created with
    // the default or user-provided optimization level.
    auto llvmTransformer =
        mlir::makeOptimizingTransformer(mlirOptLevel, /*sizeLevel=*/0, targetMachine.get());
    auto maybeEngine = mlir::ExecutionEngine::create(m_module.get(), llvmTransformer, mlirOptLevel);
    NGRAPH_CHECK(maybeEngine, "failed to construct an execution engine");
    m_engine = std::move(maybeEngine.get());
}

/// Returns the cache level size from `targetInfo` for the `cacheLevel` provided. If `userCacheSize`
/// is not zero, it returns `userCacheSize`.
static unsigned getCacheLevelSize(llvm::TargetTransformInfo& targetInfo,
                                  unsigned cacheLevel,
                                  unsigned userCacheSize)
{
    if (userCacheSize)
    {
        return userCacheSize;
    }

    llvm::Optional<unsigned> optCacheLevelSize;
    switch (cacheLevel)
    {
    case 1:
        optCacheLevelSize = targetInfo.getCacheSize(llvm::TargetTransformInfo::CacheLevel::L1D);
        break;
    case 2:
        optCacheLevelSize = targetInfo.getCacheSize(llvm::TargetTransformInfo::CacheLevel::L2D);
        break;
    default:
        NGRAPH_UNREACHABLE("Unsupported cache level: ", cacheLevel, ". Only 1 and 2 are supported");
    }

    NGRAPH_CHECK(optCacheLevelSize.hasValue() && "Cache level size is not available in TTI");
    return optCacheLevelSize.getValue();
}

// Receives affine dialect as input and applies affine and standard dialect based optimizations.
// Lowering from affine dialect to standard dialect happens along the way. Output consists of
// standard dialect only ops.
void MLIRCompiler::optimize()
{
    // Create target transform info to obtain some target information to be used in MLIR
    // optimizations. This is a temporary attempt to retrieve some target information by reusing
    // LLVM TTI infra while MLIR does not have target model.
    llvm::LLVMContext llvmContext;
    auto module = std::unique_ptr<llvm::Module>(new llvm::Module("test", llvmContext));
    module->setDataLayout(targetMachine->createDataLayout());
    auto ttiSetupFunc = llvm::cast<llvm::Function>(
        module
            ->getOrInsertFunction("__ngraph_tti_setup",
                                  llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), {}))
            .getCallee());
    auto targetInfo = targetMachine->getTargetTransformInfo(*ttiSetupFunc);

    // Populate pass manager with affine dialect optimizations.
    mlir::PassManager pm(&m_context);
    if (clEnableAffineLoopFusion)
    {
        pm.addPass(mlir::createLoopFusionPass());
    }

    if (clEnableAffineLoopTiling)
    {
        unsigned cacheLevelSize =
            getCacheLevelSize(targetInfo, clLoopTilingCacheLevel, clLoopTilingCacheSize);
        LLVM_DEBUG(llvm::dbgs() << "Enabling Affine Loop Tiling for cache level "
                                << clLoopTilingCacheLevel
                                << ": "
                                << cacheLevelSize
                                << " bytes.\n");
        pm.addPass(mlir::createLoopTilingPass(cacheLevelSize));
    }

    // Populate pass manager with affine dialect to Std dialect conversion.
    pm.addPass(mlir::createLowerAffinePass());

    // Apply any generic pass manager command line options.
    mlir::applyPassManagerCLOptions(pm);

    // Run pass manager passes.
    auto result = pm.run(m_module.get());
    NGRAPH_CHECK(succeeded(result), "Affine optimizaitons and convertion to Std dialect failed");

    // Run Std dialect optimizations.
    // TODO
}

// MLIR builders
#define TI(x) std::type_index(typeid(x))

void MLIRCompiler::buildNgDialect()
{
    const NodeVector& subGraph = m_compiledKernel->get_node_list();

    for (auto np : subGraph)
    {
        auto it = opDispatcher.find(TI(*np));
        if (it == opDispatcher.end())
        {
            throw unsupported_op{std::string{"The MLIR backend doesn't currently implement the '"} +
                                 np->description() + "' operation"};
        }
        mlir::Operation* op = it->second(*this, np.get());
        // This assumes simple 1:1 mapping between output edges and generated MLIR op results
        // If the mapping is more complex, the create_op helper can return null operation
        // and handles populating the value map itself
        if (op)
        {
            for (auto i = 0; i < op->getNumResults(); i++)
            {
                mlir::Value* result = op->getResult(i);
                if (result)
                {
                    updateTensorValue(np->get_output_tensor_ptr(i).get(), result);
                }
            }
        }
    }
    createReturn();
}

namespace ngraph
{
    namespace runtime
    {
        namespace ngmlir
        {
            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Add)
            {
                return compiler.createGenericOp<mlir::NGAddOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Subtract)
            {
                return compiler.createGenericOp<mlir::NGSubOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Multiply)
            {
                return compiler.createGenericOp<mlir::NGMulOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Divide)
            {
                return compiler.createGenericOp<mlir::NGDivOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Greater)
            {
                return compiler.createGenericOp<mlir::NGGreaterOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Less)
            {
                return compiler.createGenericOp<mlir::NGLessOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Maximum)
            {
                return compiler.createGenericOp<mlir::NGMaxOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Minimum)
            {
                return compiler.createGenericOp<mlir::NGMinOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::ArgMax)
            {
                return compiler.createIndexReduction<mlir::NGArgMaxRedOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::ArgMin)
            {
                return compiler.createIndexReduction<mlir::NGArgMinRedOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Dot)
            {
                return compiler.createGenericOp<mlir::NGDotOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Concat)
            {
                auto concat = static_cast<const ngraph::op::Concat*>(ngNode);
                auto op = compiler.createGenericOp<mlir::NGConcatOp>(ngNode);
                op->setAttr(
                    "concatenation_axis",
                    compiler.m_builder->getI64IntegerAttr(concat->get_concatenation_axis()));
                return op;
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Gather)
            {
                auto gather = static_cast<const ngraph::op::Gather*>(ngNode);
                auto op = compiler.createGenericOp<mlir::NGGatherOp>(ngNode);
                op->setAttr("axis", compiler.m_builder->getI64IntegerAttr(gather->get_axis()));
                return op;
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Relu)
            {
                return compiler.createGenericOp<mlir::NGReluOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Negative)
            {
                return compiler.createGenericOp<mlir::NGNegOp>(ngNode);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Convolution)
            {
                mlir::Operation* op = compiler.createGenericOp<mlir::NGConvolutionOp>(ngNode);
                auto convNode = static_cast<const ngraph::op::Convolution*>(ngNode);
                auto convOp = llvm::cast<mlir::NGConvolutionOp>(op);

                mlir::ArrayAttr attr =
                    compiler.getShapeAsAttr(convNode->get_window_movement_strides());
                convOp.setStrides(attr);

                attr = compiler.getShapeAsAttr(convNode->get_padding_below());
                convOp.setPadBelow(attr);

                attr = compiler.getShapeAsAttr(convNode->get_padding_above());
                convOp.setPadAbove(attr);
                return op;
            }
        }
    }
}

template <typename Op>
mlir::Operation* MLIRCompiler::createGenericOp(const ngraph::Node* ngNode)
{
    std::vector<mlir::Value*> argValues;
    std::vector<mlir::Type> resTypes;
    auto inputMap = m_compiledKernel->get_input_map();
    std::shared_ptr<descriptor::Tensor> argTensor;
    for (auto& argOutput : ngNode->input_values())
    {
        auto argOutputNode = argOutput.get_node();
        if (as_type<op::Parameter>(argOutputNode))
        {
            auto it = inputMap.find(argOutputNode->shared_from_this());
            NGRAPH_CHECK(it != inputMap.end(), "Parameter not in CK input map");

            argTensor = m_compiledKernel->input_values().at(it->second).get_tensor_ptr();
        }
        else
        {
            argTensor = argOutput.get_tensor_ptr();
        }

        auto argV = getTensorValue(argTensor.get()).m_value;
        argValues.push_back(argV);
    }

    for (auto& output : ngNode->outputs())
    {
        resTypes.push_back(getMlirType(output.get_tensor_ptr().get()));
    }

    return (m_builder->create<Op,
                              ArrayRef<mlir::Type>,
                              ArrayRef<mlir::Value*>,
                              ArrayRef<mlir::NamedAttribute>>(
                mlir::UnknownLoc::get(&m_context), resTypes, argValues, {/* no attrs */}))
        .getOperation();
}

const MLIRCompiler::MLIRCompOpMap MLIRCompiler::opDispatcher{
#define MLIR_OP(OP) {TI(ngraph::op::OP), &MLIRCompiler::createOp<ngraph::op::OP>},
#include "ops_supported.inc"
};

void MLIRCompiler::createReturn()
{
    std::vector<mlir::Value*> valueList;
    for (auto output : m_compiledKernel->get_kernel_outputs())
    {
        valueList.push_back(getTensorValue(output->get_output_tensor_ptr().get()).m_value);
    }
    m_builder->create<mlir::NGReturnOp>(mlir::UnknownLoc::get(&m_context), valueList);
}

template <typename RedOp>
mlir::Operation* MLIRCompiler::createIndexReduction(const ngraph::Node* ngNode)
{
    auto* idxRed = static_cast<const ngraph::op::util::IndexReduction*>(ngNode);
    auto op = createGenericOp<RedOp>(ngNode);
    mlir::ArrayAttr redAxesAttr =
        m_builder->getI64ArrayAttr({(int64_t)idxRed->get_reduction_axis()});
    op->setAttr("axes", redAxesAttr);
    return op;
}

void MLIRCompiler::optimizeNgDialect()
{
    mlir::PassManager pm(&m_context);
    mlir::applyPassManagerCLOptions(pm);
    if (clEnableNgInPlaceMemoryOpt)
    {
        pm.addPass(mlir::createMemoryOptimizationPass());
    }

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }
}

// Binds MLIR function arguments to the proper values. This includes externally allocated tensors
// helpers to be used inside the function.
void MLIRCompiler::bindArguments(std::vector<void*>& externalTensors)
{
    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    mlir::FuncOp func = m_module->lookupSymbol<mlir::FuncOp>("main");
    NGRAPH_CHECK(func && !func.getBlocks().empty(), "Function not found");

    // Set external arguments
    NGRAPH_CHECK(m_compiledKernel, "No compiled kernel set for compiler");
    NGRAPH_CHECK((m_compiledKernel->get_arguments().size() +
                  m_compiledKernel->get_kernel_outputs().size()) == externalTensors.size(),
                 "Number of arguments and outputs doesn't match number of tensors");
    m_externalTensors = &externalTensors;

    // Create list with a type-erased double pointer for each invocation arguments.
    // We currently use 'allocateMemrefArgs', which creates the arguments list per call ABI (see
    // comment below).
    // StaticFloatMemref is just a struct with the actual pointer to the data.

    auto expectedArguments = allocateMemrefArgs();
    NGRAPH_CHECK(expectedArguments.size(), "Arguments can't be created");
    m_invokeArgs = std::move(expectedArguments);

    NGRAPH_CHECK(m_invokeArgs.size() == m_externalTensors->size(),
                 "Number of external tensors doesn't match number of function arguments");

    // Assign external tensor pointers to invocation arguments.
    for (size_t i = 0, numArgs = m_invokeArgs.size(); i < numArgs; ++i)
    {
        auto* memRefArg = *(reinterpret_cast<mlir::StaticFloatMemRef**>(m_invokeArgs[i]));
        memRefArg->data = reinterpret_cast<float*>((*m_externalTensors)[i]);
    }
}

// Lowers standard dialect to LLVM dialect and uses the MLIR execution engine to execute the code.
void MLIRCompiler::execute()
{
    // Invoke the JIT-compiled function with the arguments. Note that, for API
    // uniformity reasons, it takes a list of type-erased pointers to arguments.
    // Please, note that 'invoke' method is overloaded with a parameter pack version.
    // Make sure the MutableArrayRef version is invoked.
    auto invocationResult = m_engine->invoke("main", llvm::MutableArrayRef<void*>(m_invokeArgs));

    if (clDumpObjectFile)
    {
        m_engine->dumpToObjectFile(clObjectFilename.empty() ? "jitted_mlir.o"
                                                            : clObjectFilename.getValue());
    }
    NGRAPH_CHECK(!invocationResult, "JIT invocation of 'main' failed\n");
}

void MLIRCompiler::cleanup()
{
    // Free void double pointer arguments without freeing external tensor data.
    for (auto* arg : m_invokeArgs)
    {
        auto* memRefArg = *(reinterpret_cast<mlir::StaticFloatMemRef**>(arg));
        free(memRefArg);
        free(arg);
    }

    // Free MLIR function builder.
    if (m_builder)
    {
        m_builder.reset(nullptr);
    }
}

// The current call ABI takes a single arg pointer (argPtr) pointing to a list of args.
// Each arg is a  pointer to a StaticFloatMemRef which contains a data pointer
//
// The args are laid out as follows
// argPtr-> arg[0]-> StaticFloatMemRef -> <data>
//          arg[1]-> StaticFloatMemRef -> <data>
//          ...
SmallVector<void*, 8> MLIRCompiler::allocateMemrefArgs()
{
    SmallVector<void*, 8> args;
    for (auto i = 0; i < m_externalTensors->size(); i++)
    {
        auto descriptor = allocateMemrefDescriptor();
        mlir::StaticFloatMemRef** arg =
            reinterpret_cast<mlir::StaticFloatMemRef**>(malloc(sizeof(mlir::StaticFloatMemRef*)));
        *arg = descriptor;
        args.push_back(arg);
    }
    return args;
}

mlir::StaticFloatMemRef* MLIRCompiler::allocateMemrefDescriptor()
{
    // We only use StaticFloatMemRef because that's what MLIR currently offers.
    // We should expand this with different types and dynamic MemRefs
    auto* descriptor =
        reinterpret_cast<mlir::StaticFloatMemRef*>(malloc(sizeof(mlir::StaticFloatMemRef)));
    NGRAPH_CHECK(descriptor != nullptr, "NULL MemRef descriptor");
    descriptor->data = nullptr;
    return descriptor;
}

void MLIRCompiler::dumpMlirModule(const std::string msg)
{
    if (clPrintIRAfterAll)
    {
        llvm::dbgs() << "*** IR Dump After " << msg << " ***\n";
        m_module->dump();
        llvm::dbgs() << "\n\n";
    }
}
