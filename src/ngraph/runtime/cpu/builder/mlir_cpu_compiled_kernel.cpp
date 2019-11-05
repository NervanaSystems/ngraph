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

#include "ngraph/runtime/cpu/cpu_builder.hpp"

#include "contrib/mlir/backend/cpu/cpu_backend.hpp"
#include "contrib/mlir/core/compiler.hpp"
#include "contrib/mlir/runtime/cpu/cpu_runtime.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/runtime/cpu/cpu_runtime_context.hpp"

using namespace ngraph;
using namespace ngraph::op;
using namespace ngraph::runtime::cpu;
using namespace ngraph::runtime::ngmlir;

#define TI(x) type_index(typeid(x))

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(CompiledKernel)
            {
                auto& functors = external_function->get_functors();

                // Tensors haven't been allocated yet so we have to keep a pointer to the pointer
                // that will hold the future memory address.
                std::vector<size_t> buffer_indices;
                for (const TensorViewWrapper& arg : args)
                {
                    auto buffer_index = external_function->get_buffer_index(arg.get_name());
                    buffer_indices.push_back(buffer_index);
                }

                for (const TensorViewWrapper& result : out)
                {
                    auto buffer_index = external_function->get_buffer_index(result.get_name());
                    buffer_indices.push_back(buffer_index);
                }

                // Create functor that will be executed to compile and run this CompiledKernel.
                // Note that 'double_ptr_args' must be captured by value since it's a local var.
                auto functor = [node, buffer_indices](CPURuntimeContext* ctx,
                                                      CPUExecutionContext* ectx) {

                    // MLIR requires a list of type-erased pointer to arguments. Tensors must have
                    // been allocated at this point so we can get rid of the extra reference.
                    std::vector<void*> ptr_args;
                    for (auto& buffer_index : buffer_indices)
                    {
                        ptr_args.push_back(ctx->buffer_data[buffer_index]);
                    }
                    // Compile nodes within the CompiledKernel op.
                    CompiledKernel* compiled_kernel =
                        static_cast<CompiledKernel*>(const_cast<Node*>(node));

                    auto it = ctx->mlir_runtimes.find(compiled_kernel);

                    if (it == ctx->mlir_runtimes.end())
                    {
                        // Compile the sub-graph and create a new runtime
                        // We must create an MLIRContext that out lives the compilation/execution
                        // The runtime contains the context and gets store in the CK cache

                        // Runtime contains context and must be constructed in-place.
                        // MLIR contexts cannot be copied over
                        ctx->mlir_runtimes.emplace(std::piecewise_construct,
                                                   std::make_tuple(compiled_kernel),
                                                   std::make_tuple());
                        MLIRCPURuntime& mlir_runtime =
                            ctx->mlir_runtimes.find(compiled_kernel)->second;
                        // Grab the context and initialize a core compiler
                        mlir::MLIRContext& context = mlir_runtime.get_context();
                        MLIRCompiler mlir_compiler(compiled_kernel, context);
                        // Compile to NG dialect
                        mlir_compiler.compile();
                        // Grab a context and initialize a CPU backend using same context
                        MLIRCPUBackend mlir_backend(mlir_compiler.get_module(), context);
                        // Codegen to LLVM dialect
                        mlir_backend.codegen();
                        // Store module into runtime, and invoke.
                        mlir_runtime.set_module(mlir_backend.get_module());
                        mlir_runtime.run(&ptr_args);
                    }
                    else
                    {
                        // We have found a cached runtime, just invoke.
                        MLIRCPURuntime& mlir_runtime = it->second;
                        mlir_runtime.run(&ptr_args);
                    }
                };

                functors.emplace_back(functor);
            }
        }
    }
}

#undef TI
