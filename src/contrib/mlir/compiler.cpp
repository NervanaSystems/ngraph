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

// NOTE: This file follows nGraph format style and naming convention since it
// exposes a public API to the rest of nGraph codebase.

#include "compiler.hpp"

#include "dialect/dialect.hpp"
#include "dialect/ops.hpp"
#include "dialect/type.hpp"
#include "lowerer.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/util/index_reduction.hpp"
#include "ngraph/type/element_type.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/ControlFlowToCFG/ConvertControlFlowToCFG.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h>
#include <mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/Passes.h>

#include <memory>
#include <mutex>

using llvm::SmallVector;
using llvm::StringRef;
using llvm::make_unique;
using llvm::ArrayRef;
using namespace ngraph::runtime::ngmlir;

static llvm::cl::opt<bool>
    clEnableAffineLoopFusion("enable-affine-loop-fusion",
                             llvm::cl::init(false),
                             llvm::cl::desc("Enable loop fusion optimization in Affine dialect"));

#define COMPILE_OP_DECL(op_name)                                                                   \
    create_op<op_name>(MLIRCompiler & compiler, const ngraph::Node* ng_node)

void MLIRCompiler::init_mlir()
{
    // Mutex to safely initialize MLIR.
    static std::mutex mlir_init_mutex;
    static bool initialized = false;

    std::unique_lock<std::mutex> lock(mlir_init_mutex);

    if (!initialized)
    {
        mlir::registerDialect<mlir::NGraphOpsDialect>();
        // Register any LLVM command line options
        llvm::cl::ParseEnvironmentOptions("ngraph", "NGRAPH_MLIR_OPTIONS", "");
        initialized = true;
    }
}

void MLIRCompiler::compile()
{
    build_ng_dialect_module();
    lower_ng_dialect();
}

void MLIRCompiler::run(std::vector<void*>& external_tensors)
{
    bind_arguments(external_tensors);
    execute();
    cleanup();
}

unsigned MLIRCompiler::get_mem_mgr_arg_id(mlir::FuncOp& func)
{
    return func.getNumArguments() - 1;
}

// Creates an MLIR module and function with nGraph dialect ops from the input CompiledKernel.
void MLIRCompiler::build_ng_dialect_module()
{
    // initialize an empty module
    m_module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&m_context));

    TypeList args_type_list, result_type_list;

    // Retrieve input and output tensors.
    const auto& kernel_inputs = m_compiled_kernel->get_arguments();
    const auto& kernel_outputs = m_compiled_kernel->get_kernel_outputs();
    NGRAPH_CHECK(kernel_inputs.size() != 0, "Cannot have empty inputs list");
    NGRAPH_CHECK(kernel_outputs.size() != 0, "Cannot have empty outputs list");

    for (auto input : kernel_inputs)
    {
        args_type_list.push_back(get_mlir_type(input.get()));
    }

    for (auto output : kernel_outputs)
    {
        result_type_list.push_back(get_mlir_type(output.get()));
    }

    auto func_type = mlir::FunctionType::get(args_type_list, result_type_list, &m_context);
    auto function = mlir::FuncOp::create(mlir::UnknownLoc::get(&m_context), "main", func_type);
    function.addEntryBlock();

    // populate Tensor->Value maps
    int i = 0;
    for (auto input : kernel_inputs)
    {
        mlir::Value* arg = function.getArgument(i);
        TensorInfo tensor_info{arg};
        m_tensor_to_value_map.insert(
            TensorToInfo(input->get_output_tensor_ptr().get(), tensor_info));
        i++;
    }

    // create builder
    m_builder = llvm::make_unique<mlir::OpBuilder>(function.getBody());
    build_ng_dialect();
    m_module->push_back(function);
    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Invalid module after lowering to NG dialect");
    }

    dump_mlir_module("nGraph Dialect Dump:");
}

// Converts nGraph shape \p ng_shape to MLIR shape \p mlir_shape.
static void get_mlir_shape(ngraph::Shape ng_shape, llvm::SmallVectorImpl<int64_t>& mlir_shape)
{
    for (auto dim : ng_shape)
    {
        mlir_shape.push_back(dim);
    }
}

// Converts an nGraph Tensor into an MLIR tensor type, including the conversion of the Tensor's
// element type.
mlir::Type MLIRCompiler::get_mlir_type(const descriptor::Tensor* tensor)
{
    SmallVector<int64_t, 4> mlir_shape;
    get_mlir_shape(tensor->get_shape(), mlir_shape);
    return mlir::NGTensorType::get(
        &m_context, get_mlir_type(tensor->get_element_type()), mlir_shape);
}

// Converts an nGraph element type into an MLIR type.
mlir::Type MLIRCompiler::get_mlir_type(const element::Type& type)
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

mlir::Type MLIRCompiler::get_mlir_type(const ngraph::Node* node)
{
    descriptor::Tensor* out_tensor = node->get_output_tensor_ptr().get();
    return get_mlir_type(out_tensor);
}

void MLIRCompiler::update_tensor_value(descriptor::Tensor* tensor, mlir::Value* value)
{
    NGRAPH_CHECK(m_tensor_to_value_map.find(tensor) == m_tensor_to_value_map.end(),
                 "tensor value already defined");
    TensorInfo tensor_info{value};
    m_tensor_to_value_map.insert(TensorToInfo(tensor, tensor_info));
}

MLIRCompiler::TensorInfo MLIRCompiler::get_tensor_value(descriptor::Tensor* tensor)
{
    auto it = m_tensor_to_value_map.find(tensor);

    NGRAPH_CHECK(it != m_tensor_to_value_map.end(), "Undefined tensor");

    return it->second;
}

// Lowers nGraph dialect all the way to LLVM module.
void MLIRCompiler::lower_ng_dialect()
{
    // Lower NG dialect to Affine
    mlir::PassManager pm;
    pm.addPass(mlir::createDialectLoweringPass(this));
    pm.addPass(mlir::createCanonicalizerPass());

    pm.run(m_module.get());

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Incorrect module after dialect lowering");
    }

    dump_mlir_module("Affine Dialect Dump (Pre-Optimizations):");

    optimize();

    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    // Lower Standard dialect to LLVM dialect.
    mlir::LLVMTypeConverter llvm_converter(&m_context);
    OwningRewritePatternList patterns;
    mlir::populateLoopToStdConversionPatterns(patterns, &m_context);
    mlir::populateStdToLLVMConversionPatterns(llvm_converter, patterns);

    mlir::ConversionTarget target(m_context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp, mlir::ModuleTerminatorOp>();
    target.addDynamicallyLegalOp<mlir::FuncOp>(
        [&](mlir::FuncOp op) { return llvm_converter.isSignatureLegal(op.getType()); });
    auto result = applyFullConversion(*m_module, target, std::move(patterns), &llvm_converter);
    NGRAPH_CHECK(succeeded(result), "Standard to LLVM dialect conversion failed");

    dump_mlir_module("LLVM-IR Dialect Dump:");

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    unsigned opt_level = 3;
    if (char* opt_level_str = std::getenv("NGRAPH_MLIR_OPT_LEVEL"))
    {
        opt_level = std::stoi(opt_level_str);
        NGRAPH_CHECK(opt_level >= 0 && opt_level <= 3, "Invalid optimization level");
    }
    // Create an MLIR execution engine. We use a null MLIR pass manager for now to make sure we
    // don't run MLIR passes that were already run. We also pass a default transformer to run
    // LLVM optimizations at level 3.
    auto llvm_transformer =
        mlir::makeOptimizingTransformer(opt_level /*optLevel*/, 0 /*sizeLevel*/);
    auto maybeEngine = mlir::ExecutionEngine::create(m_module.get(), llvm_transformer);
    NGRAPH_CHECK(maybeEngine, "failed to construct an execution engine");
    m_engine = std::move(maybeEngine.get());
}

void MLIRCompiler::optimize()
{
    // Run Affine dialect optimizations.
    mlir::PassManager pm_opts;
    if (clEnableAffineLoopFusion)
    {
        pm_opts.addPass(mlir::createLoopFusionPass());
    }

    auto opt_res = pm_opts.run(m_module.get());
    NGRAPH_CHECK(succeeded(opt_res), "Affine optimizations failed");
    dump_mlir_module("Affine Dialect Dump (Post-Optimizations):");

    // Run Affine dialect to Std dialect conversion.
    mlir::PassManager pm_lowering;
    pm_lowering.addPass(mlir::createLowerAffinePass());
    auto lowering_res = pm_lowering.run(m_module.get());
    NGRAPH_CHECK(succeeded(lowering_res), "Affine convertion to Std dialect failed");
    dump_mlir_module("Standard Dialect Dump:");

    // Run Std dialect optimizations.
    // TODO
}

// MLIR builders
#define TI(x) std::type_index(typeid(x))

void MLIRCompiler::build_ng_dialect()
{
    const NodeVector& sub_graph = m_compiled_kernel->get_node_list();

    for (auto np : sub_graph)
    {
        auto it = op_dispatcher.find(TI(*np));
        if (it == op_dispatcher.end())
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
                    update_tensor_value(np->get_output_tensor_ptr(i).get(), result);
                }
            }
        }
    }
    create_return();
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
                return compiler.create_generic_op<mlir::NGAddOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Subtract)
            {
                return compiler.create_generic_op<mlir::NGSubOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Multiply)
            {
                return compiler.create_generic_op<mlir::NGMulOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Divide)
            {
                return compiler.create_generic_op<mlir::NGDivOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Greater)
            {
                return compiler.create_generic_op<mlir::NGGreaterOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Less)
            {
                return compiler.create_generic_op<mlir::NGLessOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Maximum)
            {
                return compiler.create_generic_op<mlir::NGMaxOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Minimum)
            {
                return compiler.create_generic_op<mlir::NGMinOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::ArgMax)
            {
                return compiler.create_index_reduction<mlir::NGArgMaxRedOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::ArgMin)
            {
                return compiler.create_index_reduction<mlir::NGArgMinRedOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Dot)
            {
                return compiler.create_generic_op<mlir::NGDotOp>(ng_node);
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Concat)
            {
                auto ng_node_concat = static_cast<const ngraph::op::Concat*>(ng_node);
                auto op = compiler.create_generic_op<mlir::NGConcatOp>(ng_node);
                op->setAttr("concatenation_axis",
                            compiler.m_builder->getI64IntegerAttr(
                                ng_node_concat->get_concatenation_axis()));
                return op;
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Gather)
            {
                auto ng_node_gather = static_cast<const ngraph::op::Gather*>(ng_node);
                auto op = compiler.create_generic_op<mlir::NGGatherOp>(ng_node);
                op->setAttr("axis",
                            compiler.m_builder->getI64IntegerAttr(ng_node_gather->get_axis()));
                return op;
            }

            template <>
            mlir::Operation* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Relu)
            {
                return compiler.create_generic_op<mlir::NGReluOp>(ng_node);
            }
        }
    }
}

template <typename Op>
mlir::Operation* MLIRCompiler::create_generic_op(const ngraph::Node* ng_node)
{
    std::vector<mlir::Value*> arg_values;
    std::vector<mlir::Type> res_types;
    for (auto& arg : ng_node->get_arguments())
    {
        auto arg_tensor = arg->get_output_tensor_ptr();
        auto arg_v = get_tensor_value(arg_tensor.get()).m_value;
        arg_values.push_back(arg_v);
    }

    for (auto& output : ng_node->outputs())
    {
        res_types.push_back(get_mlir_type(output.get_tensor_ptr().get()));
    }

    return (m_builder->create<Op,
                              ArrayRef<mlir::Type>,
                              ArrayRef<mlir::Value*>,
                              ArrayRef<mlir::NamedAttribute>>(
                mlir::UnknownLoc::get(&m_context), res_types, arg_values, {/* no attrs */}))
        .getOperation();
}

const MLIRCompiler::MLIRCompOpMap MLIRCompiler::op_dispatcher{
#define MLIR_OP(OP) {TI(ngraph::op::OP), &MLIRCompiler::create_op<ngraph::op::OP>},
#include "ops_supported.inc"
};

void MLIRCompiler::create_return()
{
    std::vector<mlir::Value*> value_list;
    for (auto output : m_compiled_kernel->get_kernel_outputs())
    {
        value_list.push_back(get_tensor_value(output->get_output_tensor_ptr().get()).m_value);
    }
    m_builder->create<mlir::NGReturnOp>(mlir::UnknownLoc::get(&m_context), value_list);
}

template <typename RedOp>
mlir::Operation* MLIRCompiler::create_index_reduction(const ngraph::Node* ng_node)
{
    auto* idx_red = static_cast<const ngraph::op::util::IndexReduction*>(ng_node);
    auto op = create_generic_op<RedOp>(ng_node);
    mlir::ArrayAttr red_axes_attr =
        m_builder->getI64ArrayAttr({(int64_t)idx_red->get_reduction_axis()});
    op->setAttr("axes", red_axes_attr);
    return op;
}
// Binds MLIR function arguments to the proper values. This includes externally allocated tensors
// helpers to be used inside the function.
void MLIRCompiler::bind_arguments(std::vector<void*>& external_tensors)
{
    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    mlir::FuncOp func = m_module->lookupSymbol<mlir::FuncOp>("main");
    NGRAPH_CHECK(func && !func.getBlocks().empty(), "Function not found");

    // Set external arguments
    NGRAPH_CHECK(m_compiled_kernel, "No compiled kernel set for compiler");
    NGRAPH_CHECK((m_compiled_kernel->get_arguments().size() +
                  m_compiled_kernel->get_kernel_outputs().size()) == external_tensors.size(),
                 "Number of arguments and outputs doesn't match number of tensors");
    m_external_tensors = &external_tensors;

    // Create list with a type-erased double pointer for each invocation arguments.
    // We currently use 'allocateMemRefArguments', which creates a
    // SmallVector<StaticFloatMemref*>. StaticFloatMemref is just a struct with the
    // actual pointer to the data.

    // create MemRef args
    auto expected_arguments = allocate_memref_args();
    NGRAPH_CHECK(expected_arguments.size(), "Arguments can't be created");
    m_invoke_args = std::move(expected_arguments);

    NGRAPH_CHECK(m_invoke_args.size() == m_external_tensors->size(),
                 "Number of external tensors doesn't match number of function arguments");

    // Assign external tensor pointers to invocation arguments.
    for (size_t i = 0, num_args = m_invoke_args.size(); i < num_args; ++i)
    {
        ((mlir::StaticFloatMemRef*)m_invoke_args[i])->data = (float*)(*m_external_tensors)[i];
    }

    // Add pointer to memory manager
    // malloc here since that's what allocateMemRefArguments use
    // TODO (nmostafa): Better way of doing this ? Use builder allocator ?
    MLIRMemMgr** mem_mgr_arg = reinterpret_cast<MLIRMemMgr**>(malloc(sizeof(void*)));
    NGRAPH_CHECK(mem_mgr_arg != nullptr);
    *mem_mgr_arg = &get_mem_mgr();
    // inserting memory manager ptr in right location ?
    NGRAPH_CHECK(m_invoke_args.size() == get_mem_mgr_arg_id(func));
    m_invoke_args.push_back(static_cast<void*>(mem_mgr_arg));
}

// Lowers standard dialect to LLVM dialect and uses the MLIR execution engine to execute the code.
void MLIRCompiler::execute()
{
    // Invoke the JIT-compiled function with the arguments. Note that, for API
    // uniformity reasons, it takes a list of type-erased pointers to arguments.
    // Please, note that 'invoke' method is overloaded with a parameter pack version.
    // Make sure the MutableArrayRef version is invoked.
    auto invocationResult = m_engine->invoke("main", llvm::MutableArrayRef<void*>(m_invoke_args));
    NGRAPH_CHECK(!invocationResult, "JIT invocation of 'main' failed\n");
}

void MLIRCompiler::cleanup()
{
    // Free void double pointer arguments without freeing external tensor data.
    for (auto* arg : m_invoke_args)
    {
        free(arg);
    }

    // Free MLIR function builder.
    if (m_builder)
    {
        m_builder.reset(nullptr);
    }

    // Free allocated memory for JIT'ed code temps
    m_mem_mgr.freeAll();
}

SmallVector<void*, 8> MLIRCompiler::allocate_memref_args()
{
    SmallVector<void*, 8> args;
    for (auto i = 0; i < m_external_tensors->size(); i++)
    {
        auto descriptor = allocate_memref_descriptor();
        args.push_back(descriptor);
    }
    return args;
}

mlir::StaticFloatMemRef* MLIRCompiler::allocate_memref_descriptor()
{
    // We only use StaticFloatMemRef because that's what MLIR currently offers.
    // We should expand this with different types and dynamic MemRefs
    auto* descriptor =
        reinterpret_cast<mlir::StaticFloatMemRef*>(malloc(sizeof(mlir::StaticFloatMemRef)));
    NGRAPH_CHECK(descriptor != nullptr, "NULL MemRef descriptor");
    descriptor->data = nullptr;
    return descriptor;
}

void MLIRCompiler::dump_mlir_module(const std::string msg)
{
    if (std::getenv("NGRAPH_MLIR_DUMP_ALL") != nullptr)
    {
        llvm::dbgs() << "*** " << msg << " ***\n";
        m_module->dump();
        llvm::dbgs() << "\n\n";
    }
}
