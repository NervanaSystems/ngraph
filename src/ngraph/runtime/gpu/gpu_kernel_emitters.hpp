/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#pragma once

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_view_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            namespace kernel
            {
                void emit_prologue(codegen::CodeWriter& writer, const Node* node)
                {
                    writer << "{   //" << node->get_name() << "\n";
                    writer.indent++;
                }

                void emit_epilogue(codegen::CodeWriter& writer)
                {
                    writer.indent--;
                    writer << "}\n";
                }

                void emit_memcpyDtD(codegen::CodeWriter& writer,
                                    const GPU_TensorViewWrapper& dst,
                                    const GPU_TensorViewWrapper& src)
                {
                    writer << "runtime::gpu::cuda_memcpyDtD(" << dst.get_name() << ", "
                           << src.get_name() << ", " << dst.get_size() << " * "
                           << dst.get_element_type().size() << ");\n";
                    return;
                }
            }
        }
    }
}
