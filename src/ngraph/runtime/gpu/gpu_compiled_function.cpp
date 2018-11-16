//*****************************************************************************
// Copyright 2017-2018 Intel Corporation
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

#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <fstream>
#include <mutex>
#include <string>
#include <tuple>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_backend.hpp"
#include "ngraph/runtime/gpu/gpu_compiled_function.hpp"

using namespace std;
using namespace ngraph;

const std::string runtime::gpu::GPU_CompiledFunction::s_output_dir = "gpu_codegen";
// static std::mutex s_compilation;

class GPUStaticInitializers
{
public:
    GPUStaticInitializers()
    {
        file_util::remove_directory(runtime::gpu::GPU_CompiledFunction::s_output_dir);
        file_util::make_directory(runtime::gpu::GPU_CompiledFunction::s_output_dir);
    }
};

static GPUStaticInitializers s_static_initializers;

const size_t runtime::gpu::GPU_CompiledFunction::GPU_CompiledFunction::s_memory_pool_alignment = 64;

runtime::gpu::GPU_CompiledFunction::GPU_CompiledFunction(
    const shared_ptr<ngraph::Function>& function,
    std::shared_ptr<GPU_Backend::BackendContext>& shared_context)
    : m_compiled_function(nullptr)
    , m_function(function)
    , m_emit_timing(false)
    , m_is_compiled(false)
    , m_shared_context(shared_context)
{
}

runtime::gpu::GPU_CompiledFunction::~GPU_CompiledFunction()
{
}

// void runtime::gpu::GPU_CompiledFunction::compile()
// {
//     if (m_is_compiled)
//     {
//         return;
//     }
//     std::unique_lock<std::mutex> lock(s_compilation);

//     m_function_name = m_function->get_name();

//     auto allocator = std::make_shared<runtime::gpu::GPUAllocator>(
//         m_shared_context->m_primitive_emitter->get_memory_allocator());

//     ngraph::pass::Manager pass_manager;
// #if CUDNN_VERSION >= 7200
//     // recurrent network fusion
//     pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
//     pass_manager.register_pass<runtime::gpu::pass::RNNFusion>();
//     pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
//     pass_manager.register_pass<runtime::gpu::pass::MultiLayerRNNFusion>();
// #else
//     pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
// #endif
//     pass_manager.register_pass<runtime::gpu::pass::BatchNormCache>();
//     pass_manager.register_pass<ngraph::pass::LikeReplacement>();
//     pass_manager.register_pass<runtime::gpu::pass::GPULayout>(this);
//     pass_manager.register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
//     pass_manager.register_pass<ngraph::pass::Liveness>();
//     pass_manager.register_pass<ngraph::pass::MemoryLayout>(s_memory_pool_alignment);
//     pass_manager.register_pass<runtime::gpu::pass::TensorMemoryReservation>(
//         *allocator, m_tensor_memory_buffers);
//     std::string common_function_string;
//     auto femitter = bind(&ngraph::runtime::gpu::GPU_CompiledFunction::emit_op_as_function,
//                          this,
//                          placeholders::_1,
//                          placeholders::_2);
//     pass_manager.register_pass<ngraph::pass::CommonFunctionCollection>(
//         femitter, m_node_function_map, common_function_string);
//     string dump_filename = file_util::path_join(s_output_dir, m_function_name + "_ops.txt");
//     pass_manager.register_pass<ngraph::pass::DumpSorted>(dump_filename);

//     pass_manager.run_passes(m_function);

//     for (shared_ptr<Function> current_function : pass_manager.get_state().get_functions())
//     {
//         m_function_ordered_ops.emplace(current_function, current_function->get_ordered_ops());
//     }

//     emit_header();
//     emit_timer_functions();
//     emit_constant_declarations();
//     emit_function_declarations();
//     m_writer << common_function_string << "\n";
//     emit_functions();

//     // allocate device buffers for primitive arguments and workspace
//     allocator->close();
//     m_shared_context->m_primitive_emitter->allocate_primitive_memory();

//     string code = m_writer.get_code();
//     store_emitted_functions(code);

//     m_compiler.reset(new codegen::Compiler());
//     m_execution_engine.reset(new codegen::ExecutionEngine());
//     m_compiler->set_precompiled_header_source(get_pch_header_source());

//     auto codegen_module = m_compiler->compile(code);
//     if (codegen_module == nullptr)
//     {
//         throw runtime_error("Function failed to compile to bitcode");
//     }

//     m_execution_engine->add_module(codegen_module);
//     m_execution_engine->finalize();

//     m_compiled_function = m_execution_engine->find_function<EntryPoint_t>(m_function_name);
//     if (!m_compiled_function)
//     {
//         throw runtime_error("Function failed to compile");
//     }

//     m_is_compiled = true;
// }
