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

#include <algorithm>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <fstream>
#include <locale>
#include <mutex>
#include <string>
#include <tuple>

#include "ngraph/descriptor/input.hpp"
#include "ngraph/descriptor/layout/dense_tensor_layout.hpp"
#include "ngraph/descriptor/output.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/convert_opset_1_to_0.hpp"
#include "ngraph/pass/convert_opset_3_to_1.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/implicit_broadcast_elimination.hpp"
#include "ngraph/pass/like_replacement.hpp"

#include "gpu_backend.hpp"
#include "gpu_compiled_function.hpp"
#include "gpu_external_function.hpp"
#include "gpu_internal_function.hpp"
#include "op/batch_norm.hpp"
#include "op/rnn.hpp"
#include "pass/gpu_batch_norm_cache.hpp"
#include "pass/gpu_layout.hpp"
#include "pass/gpu_rnn_fusion.hpp"
#include "pass/tensor_memory_reservation.hpp"

using namespace std;
using namespace ngraph;

std::string runtime::gpu::GPUCompiledFunction::get_output_dir()
{
    static std::string output_dir = "gpu_codegen";
    return output_dir;
}

size_t runtime::gpu::GPUCompiledFunction::get_memory_alignment()
{
    static size_t memory_pool_alignment = 64;
    return memory_pool_alignment;
}

static std::mutex s_compilation;

class GPUStaticInitializers
{
public:
    GPUStaticInitializers()
    {
        file_util::remove_directory(runtime::gpu::GPUCompiledFunction::get_output_dir());
        file_util::make_directory(runtime::gpu::GPUCompiledFunction::get_output_dir());
    }
};

static GPUStaticInitializers s_static_initializers;

runtime::gpu::GPUCompiledFunction::GPUCompiledFunction(
    const shared_ptr<ngraph::Function>& function,
    const std::shared_ptr<GPUBackend::BackendContext>& shared_context)
    : m_runtime(nullptr)
    , m_function(function)
    , m_emit_timing(false)
    , m_is_compiled(false)
    , m_shared_context(shared_context)
{
}

runtime::gpu::GPUCompiledFunction::~GPUCompiledFunction() {}

std::vector<std::string> get_case_variants(std::vector<std::string> cases)
{
    std::vector<std::string> results;
    for (auto& c : cases)
    {
        results.push_back(c);
        if (std::all_of(c.begin(), c.end(), ::isdigit))
        {
            continue;
        }
        for (auto i = 0u; i < c.size(); i++)
        {
            c[i] = std::toupper(c[i], std::locale());
            if (i == 0)
            {
                results.emplace_back(c);
            }
        }
        results.emplace_back(c);
    }
    return results;
}

std::shared_ptr<runtime::gpu::GPUCompiledFunction> runtime::gpu::GPUCompiledFunction::make(
    const std::shared_ptr<ngraph::Function>& function,
    const std::shared_ptr<GPUBackend::BackendContext>& shared_context)
{
    return std::make_shared<runtime::gpu::GPUInternalFunction>(function, shared_context);
}

void runtime::gpu::GPUCompiledFunction::compile()
{
    if (m_is_compiled)
    {
        return;
    }
    std::unique_lock<std::mutex> lock(s_compilation);

    m_function_name = m_function->get_name();

    auto allocator = std::make_shared<runtime::gpu::GPUAllocator>(
        m_shared_context->m_primitive_emitter->get_memory_allocator());

    ngraph::pass::Manager pass_manager;
#if CUDNN_VERSION >= 7200
    // recurrent network fusion
    pass_manager.register_pass<runtime::gpu::pass::LSTMFusion>();
    pass_manager.register_pass<runtime::gpu::pass::RNNFusion>();
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
    pass_manager.register_pass<runtime::gpu::pass::MultiLayerRNNFusion>();
#else
    pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
#endif
    pass_manager.register_pass<runtime::gpu::pass::BatchNormCache>();
    pass_manager.register_pass<ngraph::pass::LikeReplacement>();
    pass_manager.register_pass<ngraph::pass::ConvertOpset3To1>();
    pass_manager.register_pass<ngraph::pass::ConvertOpset1To0>();
    pass_manager.register_pass<ngraph::pass::FusedOpDecomposition>();
    pass_manager.register_pass<ngraph::pass::ImplicitBroadcastElimination>();
    pass_manager.register_pass<runtime::gpu::pass::GPULayout>(this);
    pass_manager.register_pass<ngraph::pass::AssignLayout<descriptor::layout::DenseTensorLayout>>();
    pass_manager.register_pass<ngraph::pass::Liveness>();
    pass_manager.register_pass<ngraph::pass::MemoryLayout>(get_memory_alignment());
    pass_manager.register_pass<runtime::gpu::pass::TensorMemoryReservation>(
        *allocator, m_tensor_memory_buffers);
    string dump_filename = file_util::path_join(get_output_dir(), m_function_name + "_ops.txt");
    pass_manager.register_pass<ngraph::pass::DumpSorted>(dump_filename);
    pass_manager.run_passes(m_function);
    m_function_ordered_ops.emplace(m_function, m_function->get_ordered_ops());

    add_passes(pass_manager);
    emit();

    // allocate device buffers for primitive arguments and workspace
    allocator->close();
    m_shared_context->m_primitive_emitter->allocate_primitive_memory();

    compile_function();
    m_is_compiled = true;
}
