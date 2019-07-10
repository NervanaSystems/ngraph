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

#include "compiler.hpp"

#include "dialect/dialect.hpp"
#include "dialect/ops.hpp"
#include "dialect/type.hpp"
#include "lowerer.hpp"
#include "ngraph/descriptor/tensor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/type/element_type.hpp"

#include <llvm/ADT/STLExtras.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
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

using namespace ngraph::runtime::ngmlir;

#define COMPILE_OP_DECL(op_name)                                                                   \
    create_op<op_name>(MLIRCompiler & compiler, const ngraph::Node* ng_node)

MLIRCompiler::MLIRCompiler(const ngraph::op::CompiledKernel* compiled_kernel,
                           const std::vector<void*>& external_tensors)
    : m_compiled_kernel(compiled_kernel)
    , m_external_tensors(external_tensors)
{
    NGRAPH_CHECK((m_compiled_kernel->get_arguments().size() +
                  m_compiled_kernel->get_kernel_outputs().size()) == external_tensors.size(),
                 "Number of arguments and outputs doesn't match number of tensors");
}

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
        llvm::cl::ParseEnvironmentOptions("ngraph", "MLIR_LLVM_OPTIONS", "");
        initialized = true;
    }
}

void MLIRCompiler::compile_and_run()
{
    build_ng_dialect_module();
    lower_ng_dialect();
    optimize();
    bind_arguments();
    execute();
    cleanup();
}

// Creates an MLIR module and function with nGraph dialect ops from the input CompiledKernel.
void MLIRCompiler::build_ng_dialect_module()
{
    // initialize an empty module
    m_module = make_unique<mlir::Module>(&m_context);

    TypeList args_type_list, result_type_list;

    // Retrieve input and output tensors.
    const auto& kernel_inputs = m_compiled_kernel->get_arguments();
    const auto& kernel_outputs = m_compiled_kernel->get_kernel_outputs();
    NGRAPH_CHECK(kernel_inputs.size() != 0, "Cannot have empty inputs list");
    NGRAPH_CHECK(kernel_outputs.size() != 0, "Cannot have empty outputs list");

    for (auto input : kernel_inputs)
    {
        args_type_list.push_back(get_mlir_type(input->get_output_tensor_ptr().get()));
    }

    for (auto output : kernel_outputs)
    {
        result_type_list.push_back(get_mlir_type(output->get_output_tensor_ptr().get()));
    }

    auto func_type = mlir::FunctionType::get(args_type_list, result_type_list, &m_context);
    auto function =
        make_unique<mlir::Function>(mlir::UnknownLoc::get(&m_context), "main", func_type);
    function->addEntryBlock();

    // populate Tensor->Value maps
    int i = 0;
    for (auto input : kernel_inputs)
    {
        mlir::Value* arg = function->getArgument(i);
        TensorInfo tensor_info{arg};
        m_tensor_to_value_map.insert(
            TensorToInfo(input->get_output_tensor_ptr().get(), tensor_info));
        i++;
    }

    // create builder
    m_builder = llvm::make_unique<mlir::OpBuilder>(function->getBody());
    build_ng_dialect();
    m_module->getFunctions().push_back(function.release());
    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Invalid module after lowering to NG dialect");
    }

    dump_mlir_module("nGraph Dialect Dump:");
}

// Converts an nGraph Tensor into an MLIR tensor type, including the conversion of the Tensor's
// element type.
mlir::Type MLIRCompiler::get_mlir_type(const descriptor::Tensor* tensor)
{
    SmallVector<int64_t, 4> shape;
    for (auto d : tensor->get_shape())
    {
        shape.push_back(d);
    }

    return mlir::NGTensorType::get(&m_context, get_mlir_type(tensor->get_element_type()), shape);
}

// Converts an nGraph element type into an MLIR type.
mlir::Type MLIRCompiler::get_mlir_type(const element::Type& type)
{
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif

    switch (type.get_type_enum())
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

#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
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

// Lowers nGraph dialect to affine dialect.
void MLIRCompiler::lower_ng_dialect()
{
    mlir::PassManager pm;
    pm.addPass(mlir::createDialectLoweringPass(this));
    pm.addPass(mlir::createCanonicalizerPass());

    pm.run(m_module.get());

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Incorrect module after dialect lowering");
    }

    dump_mlir_module("Affine Dialect Dump:");
}

// Receives affine dialect as input and applies affine and standard dialect based optimizations.
// Lowering from affine dialect to standard dialect happens along the way. Output consists of
// standard dialect only ops.
void MLIRCompiler::optimize()
{
    mlir::PassManager pm;
    // Lower affine ops
    pm.addPass(mlir::createLowerAffinePass());
    auto rr = pm.run(m_module.get());
    NGRAPH_CHECK(succeeded(rr), "Affine loop lowering failed");

    dump_mlir_module("Standard Dialect Dump:");
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
        mlir::Value* mlir_value = it->second(*this, np.get());
        // builders that have multiple result values will update the value map, and set their ret values to null
        if (mlir_value)
        {
            update_tensor_value(np->get_output_tensor_ptr().get(), mlir_value);
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
            mlir::Value* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Add)
            {
                return compiler.create_binary_op<mlir::NGAddOp>(ng_node);
            }

            template <>
            mlir::Value* MLIRCompiler::COMPILE_OP_DECL(ngraph::op::Dot)
            {
                return compiler.create_binary_op<mlir::NGDotOp>(ng_node);
            }
        }
    }
}

const MLIRCompiler::MLIRCompOpMap MLIRCompiler::op_dispatcher{
#define MLIR_OP(OP) {TI(ngraph::op::OP), &MLIRCompiler::create_op<ngraph::op::OP>},
#include "ops_supported.inc"
};

template <typename BinOp>
mlir::Value* MLIRCompiler::create_binary_op(const ngraph::Node* ng_node)
{
    auto lhs = ng_node->get_argument(0)->get_output_tensor_ptr();
    auto rhs = ng_node->get_argument(1)->get_output_tensor_ptr();
    auto lhs_v = get_tensor_value(lhs.get()).m_value;
    auto rhs_v = get_tensor_value(rhs.get()).m_value;
    auto res_type = get_mlir_type(ng_node->get_output_tensor_ptr().get());
    return m_builder->create<BinOp>(mlir::UnknownLoc::get(&m_context), res_type, lhs_v, rhs_v)
        .getResult();
}

void MLIRCompiler::create_return()
{
    std::vector<mlir::Value*> value_list;
    for (auto output : m_compiled_kernel->get_kernel_outputs())
    {
        value_list.push_back(get_tensor_value(output->get_output_tensor_ptr().get()).m_value);
    }
    m_builder->create<mlir::NGReturnOp>(mlir::UnknownLoc::get(&m_context), value_list);
}

// Binds MLIR function arguments to the proper values. This includes externally allocated tensors
// helpers to be used inside the function.
void MLIRCompiler::bind_arguments()
{
    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    mlir::Function* func = m_module->getNamedFunction("main");
    NGRAPH_CHECK(func && !func->getBlocks().empty(), "Function not found");

    // Create list with a type-erased double pointer for each invocation arguments.
    // We currently use 'allocateMemRefArguments', which creates a
    // SmallVector<StaticFloatMemref*>. StaticFloatMemref is just a struct with the
    // actual pointer to the data.

    // create MemRef args
    auto expected_arguments = allocate_memref_args(func);
    NGRAPH_CHECK(expected_arguments.size(), "Arguments can't be created");
    m_invoke_args = std::move(expected_arguments);

    NGRAPH_CHECK(m_invoke_args.size() == m_external_tensors.size(),
                 "Number of external tensors doesn't match number of function arguments");

    // Assign external tensor pointers to invocation arguments.
    for (size_t i = 0, num_args = m_invoke_args.size(); i < num_args; ++i)
    {
        ((mlir::StaticFloatMemRef*)m_invoke_args[i])->data = (float*)m_external_tensors[i];
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
    NGRAPH_CHECK(m_module, "MLIR module is not ready.");

    // Lower Standard dialect to LLVM dialect.
    mlir::LLVMTypeConverter llvm_converter(&m_context);
    OwningRewritePatternList patterns;
    mlir::populateStdToLLVMConversionPatterns(llvm_converter, patterns);

    mlir::ConversionTarget target(m_context);
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();
    auto result = applyConversionPatterns(*m_module, target, llvm_converter, std::move(patterns));
    NGRAPH_CHECK(succeeded(result), "Standard to LLVM dialect conversion failed");

    dump_mlir_module("LLVM-IR Dialect Dump:");

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Create an MLIR execution engine. We use a null MLIR pass manager for now to make sure we
    // don't run MLIR passes that were already run. We also pass a default transformer to run
    // LLVM optimizations at level 3.
    auto llvm_transformer = mlir::makeOptimizingTransformer(3 /*optLevel*/, 0 /*sizeLevel*/);
    auto maybeEngine = mlir::ExecutionEngine::create(m_module.get(), llvm_transformer);
    NGRAPH_CHECK(maybeEngine, "failed to construct an execution engine");
    m_engine = std::move(maybeEngine.get());

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

SmallVector<void*, 8> MLIRCompiler::allocate_memref_args(mlir::Function* func)
{
    SmallVector<void*, 8> args;
    args.reserve(func->getNumArguments());
    for (const auto& arg : func->getArguments())
    {
        auto descriptor = allocate_memref_descriptor(arg->getType());

        if (!descriptor)
        {
            continue;
        }
        args.push_back(descriptor);
    }
    return args;
}

mlir::StaticFloatMemRef* MLIRCompiler::allocate_memref_descriptor(mlir::Type type)
{
    auto memRefType = type.dyn_cast<mlir::MemRefType>();
    if (!memRefType)
    {
        return nullptr;
    }
    NGRAPH_CHECK(memRefType.getNumDynamicDims() == 0, "No support for dynamic shapes");

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
