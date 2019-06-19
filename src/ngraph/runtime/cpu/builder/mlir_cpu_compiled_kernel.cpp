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

#include "contrib/mlir/compiler.hpp"
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
                    auto* compiled_kernel = static_cast<const CompiledKernel*>(node);

                    MLIRCompiler mlir_compiler(compiled_kernel, ptr_args);
                    // TODO: Decouple 'compile' and 'run' APIs. We want to be able to run the same
                    // jitted code on different arguments.
                    mlir_compiler.compile_and_run();
                };

                functors.emplace_back(functor);
            }
        }
    }
}

#undef TI
