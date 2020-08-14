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

#include "ngraph/runtime/mlir/mlir_executable.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "Conversion/NgraphToStandard/NgraphToStandard.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/ToolUtilities.h"
// #include "mlir/Conversion/ToolUtilities.h"
#include "mlir/Translation.h"
#include "ngraph/chrome_trace.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/except.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/pass/assign_layout.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/like_replacement.hpp"
#include "ngraph/pass/liveness.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/opset0_downgrade.hpp"
#include "ngraph/pass/opset1_downgrade.hpp"
#include "ngraph/pass/zero_dim_tensor_elimination.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/mlir/mlir_ngraph_convert.hpp"
#include "ngraph/runtime/mlir/mlir_ngraph_ops.hpp"
#include "ngraph/util.hpp"

#include "Dialect/Ngraph/NgraphDialect.h"

using namespace std;
using namespace ngraph;

using descriptor::layout::DenseTensorLayout;

runtime::mlir::MlirExecutable::MlirExecutable(const shared_ptr<Function>& function,
                                              bool enable_performance_collection)
{
    ::mlir::registerAllDialects();
    ::mlir::registerAllTranslations();

    ::mlir::registerDialect<::mlir::ngraph::NgraphDialect>();

    // ngmlir::MLIRCompiler::init();
    // ngmlir::MLIRCPUBackend::init();

    m_function = clone_function(*function);

    // auto is_supported = [](const Node& node) {
    //     bool retval = false;
    //     switch (get_typeid(node))
    //     {
    //     case OP_TYPEID::Clamp_v0:
    //     case OP_TYPEID::MatMul_v0:
    //     case OP_TYPEID::Mod_v1:
    //     case OP_TYPEID::Squeeze_v0:
    //     case OP_TYPEID::Unsqueeze_v0: retval = true; break;
    //     default: break;
    //     }
    //     return retval;
    // };
    pass::Manager pass_manager;
    pass_manager.register_pass<pass::LikeReplacement>();

    // pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    // pass_manager.register_pass<pass::FusedOpDecomposition>();

    pass_manager.register_pass<pass::Opset1Downgrade>();
    pass_manager.register_pass<pass::Opset0Downgrade>();
    // Need to decompose any v0 fused ops, which were produced by the downgrade pass
    // pass_manager.register_pass<pass::FusedOpDecomposition>(is_supported);
    pass_manager.register_pass<pass::ZeroDimTensorElimination>();
    pass_manager.run_passes(m_function);
    for (auto node : m_function->get_ordered_ops())
    {
        m_nodes.push_back(node);
    }

    set_parameters_and_results(*m_function);

    m_context.reset(new ::mlir::MLIRContext());
    m_module = NgraphToMlir::convert_function(m_function.get(), m_context.get());

    // The m_module at this point contains MLIR ngraph ops, it must be lowered to LLVM IR
    // before generating the engine
    ::mlir::PassManager pm(m_context.get());
    pm.addPass(::mlir::createLowerNgraphPass());

    // Apply any generic pass manager command line options.
    ::mlir::applyPassManagerCLOptions(pm);

    if (failed(pm.run(m_module.get())))
    {
        NGRAPH_CHECK(false, "MLIR pass manager failed");
    }

    if (failed(m_module->verify()))
    {
        NGRAPH_CHECK(false, "Invalid module after NG dialect optimization");
    }

    // dumpMlirModule("nGraph Dialect optimization", m_module.get());

    int optimization_level = 3;
    // Initialize LLVM targets and target machine for current host.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = ::mlir::makeOptimizingTransformer(optimization_level,
                                                         /*sizeLevel=*/0,
                                                         /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles the module.
    auto maybeEngine = ::mlir::ExecutionEngine::create(*m_module, optPipeline);
    assert(maybeEngine && "failed to construct an execution engine");
    NGRAPH_INFO;
    m_engine = move(maybeEngine.get());
    NGRAPH_INFO;
}

bool runtime::mlir::MlirExecutable::call(const vector<shared_ptr<runtime::Tensor>>& outputs,
                                         const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    NGRAPH_INFO;
    event::Duration d1("call", "Interpreter");

    NGRAPH_INFO;
    m_module->dump();
    NGRAPH_INFO;

    if (!m_engine)
    {
        NGRAPH_INFO;
        // init();
    }

    // if (m_first_iteration)
    // {
    //     ::mlir::MLIRContext& context = m_mlir_runtime.get_context();
    //     runtime::ngmlir::MLIRCompiler mlir_compiler(m_function, context);
    //     // Compile to NG dialect
    //     mlir_compiler.compile();
    //     // Grab a context and initialize a CPU backend using same context
    //     runtime::ngmlir::MLIRCPUBackend mlir_backend(mlir_compiler.get_module(), context);
    //     // Codegen to LLVM dialect
    //     mlir_backend.codegen();
    //     // Store module into runtime, and invoke.
    //     m_mlir_runtime.set_module(mlir_backend.get_module());
    // }

    // std::vector<runtime::ngmlir::MemRefArg> mem_ref_arg_vec;
    // for (auto tensor : inputs)
    // {
    //     auto host_tensor = dynamic_pointer_cast<runtime::HostTensor>(tensor);
    //     if (!host_tensor)
    //     {
    //         throw runtime_error("call args are not HostTensor");
    //     }
    //     runtime::ngmlir::MemRefArg mem_ref_arg;
    //     mem_ref_arg.m_tensor = host_tensor->get_data_ptr();
    //     mem_ref_arg.m_shape = tensor->get_shape();
    //     mem_ref_arg.m_strides = tensor->get_strides();
    //     mem_ref_arg_vec.push_back(mem_ref_arg);
    // }

    // // convert outputs to HostTensor
    // for (auto tensor : outputs)
    // {
    //     auto host_tensor = dynamic_pointer_cast<runtime::HostTensor>(tensor);
    //     if (!host_tensor)
    //     {
    //         throw runtime_error("call args are not HostTensor");
    //     }
    //     runtime::ngmlir::MemRefArg mem_ref_arg;
    //     mem_ref_arg.m_tensor = host_tensor->get_data_ptr();
    //     mem_ref_arg.m_shape = tensor->get_shape();
    //     mem_ref_arg.m_strides = tensor->get_strides();
    //     mem_ref_arg_vec.push_back(mem_ref_arg);
    // }

    // m_mlir_runtime.run(mem_ref_arg_vec, m_first_iteration);
    // m_first_iteration = false;

    return true;
}

shared_ptr<ngraph::op::Parameter> runtime::mlir::MlirExecutable::get_parameter(size_t index) const
{
    const ParameterVector& parameters = get_parameters();
    NGRAPH_CHECK(index < parameters.size(), "create_tensor for input out of bounds");
    return parameters[index];
}

shared_ptr<ngraph::op::Result> runtime::mlir::MlirExecutable::get_result(size_t index) const
{
    const ResultVector& results = get_results();
    NGRAPH_CHECK(index < results.size(), "create_tensor for input out of bounds");
    return results[index];
}
shared_ptr<runtime::Tensor> runtime::mlir::MlirExecutable::create_input_tensor(size_t input_index)
{
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    return make_shared<runtime::HostTensor>(parameter->get_output_element_type(0),
                                            parameter->get_output_shape(0));
}

shared_ptr<runtime::Tensor> runtime::mlir::MlirExecutable::create_output_tensor(size_t output_index)
{
    shared_ptr<op::Result> result = get_result(output_index);
    return make_shared<runtime::HostTensor>(result->get_output_element_type(0),
                                            result->get_output_shape(0));
}

vector<shared_ptr<runtime::Tensor>>
    runtime::mlir::MlirExecutable::create_input_tensor(size_t input_index, size_t pipeline_depth)
{
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Parameter> parameter = get_parameter(input_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::HostTensor> tensor;
        auto t = make_shared<runtime::HostTensor>(parameter->get_output_element_type(0),
                                                  parameter->get_output_shape(0));
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

vector<shared_ptr<runtime::Tensor>>
    runtime::mlir::MlirExecutable::create_output_tensor(size_t output_index, size_t pipeline_depth)
{
    vector<shared_ptr<runtime::HostTensor>> tensors;
    shared_ptr<op::Result> result = get_result(output_index);
    for (size_t i = 0; i < pipeline_depth; i++)
    {
        shared_ptr<runtime::HostTensor> tensor;
        auto t = make_shared<runtime::HostTensor>(result->get_output_element_type(0),
                                                  result->get_output_shape(0));
        tensor = static_pointer_cast<runtime::HostTensor>(t);
        tensors.push_back(tensor);
    }
    vector<shared_ptr<runtime::Tensor>> result_tensors;
    for (const shared_ptr<runtime::HostTensor>& tensor : tensors)
    {
        result_tensors.push_back(tensor);
    }
    return result_tensors;
}

/// Creates target machine for current host.
llvm::Expected<std::unique_ptr<llvm::TargetMachine>>
    runtime::mlir::MlirExecutable::create_default_target_machine(unsigned optLevel)
{
    auto machineBuilder = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!machineBuilder)
    {
        return machineBuilder.takeError();
    }

    // Relocation model and code model are kept to default values. CodeGen
    // optimization level
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

// int runtime::mlir::MlirExecutable::dumpLLVMIR(mlir::ModuleOp module)
// {
//     auto llvmModule = mlir::translateModuleToLLVMIR(module);
//     if (!llvmModule)
//     {
//         llvm::errs() << "Failed to emit LLVM IR\n";
//         return -1;
//     }

//     // Initialize LLVM targets.
//     llvm::InitializeNativeTarget();
//     llvm::InitializeNativeTargetAsmPrinter();
//     mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

//     /// Optionally run an optimization pipeline over the llvm module.
//     auto optPipeline = mlir::makeOptimizingTransformer(
//         /*optLevel=*/enableOpt ? 3 : 0,
//         /*sizeLevel=*/0,
//         /*targetMachine=*/nullptr);
//     if (auto err = optPipeline(llvmModule.get()))
//     {
//         llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
//         return -1;
//     }
//     llvm::errs() << *llvmModule << "\n";
//     return 0;
// }

// int runtime::mlir::MlirExecutable::runJit(mlir::ModuleOp module)
// {
//     // Initialize LLVM targets.
//     llvm::InitializeNativeTarget();
//     llvm::InitializeNativeTargetAsmPrinter();

//     // An optimization pipeline to use within the execution engine.
//     auto optPipeline = mlir::makeOptimizingTransformer(
//         /*optLevel=*/enableOpt ? 3 : 0,
//         /*sizeLevel=*/0,
//         /*targetMachine=*/nullptr);

//     // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
//     // the module.
//     auto maybeEngine = mlir::ExecutionEngine::create(module, optPipeline);
//     assert(maybeEngine && "failed to construct an execution engine");
//     auto& engine = maybeEngine.get();

//     // Invoke the JIT-compiled function.
//     auto invocationResult = engine->invoke("main");
//     if (invocationResult)
//     {
//         llvm::errs() << "JIT invocation failed\n";
//         return -1;
//     }

//     return 0;
// }

// void runtime::mlir::MlirExecutable::optimize_dialect()
// {
//     ::mlir::PassManager pm(&m_context);
//     if (clEnableNgKernelLibFusion)
//     {
//         pm.addPass(ngraph::pass::createNgDialectFusedOpsPass());
//     }

//     // Apply any generic pass manager command line options.
//     ::mlir::applyPassManagerCLOptions(pm);

//     if (failed(pm.run(m_module.get())))
//     {
//         NGRAPH_CHECK(false, "MLIR pass manager failed");
//     }

//     if (failed(m_module->verify()))
//     {
//         NGRAPH_CHECK(false, "Invalid module after NG dialect optimization");
//     }

//     dumpMlirModule("nGraph Dialect optimization", m_module.get());
// }
