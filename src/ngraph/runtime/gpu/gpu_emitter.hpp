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

#pragma once

#include <string>
#include <vector>

#include "ngraph/codegen/code_writer.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/gpu/gpu_external_function.hpp"
#include "ngraph/runtime/gpu/gpu_tensor_wrapper.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace gpu
        {
            class GPU_Emitter
            {
            public:
                static std::function<void(EMIT_ARGS)> get_emit_function(const Node& node);

// This defines a collection of function declarations like this
// static void emit_Abs(EMIT_ARGS);
// static void emit_Acos(EMIT_ARGS);
#define NGRAPH_OP(a, b) static void emit_##a(EMIT_ARGS);
#include "ngraph/runtime/gpu/op/op_tbl.hpp"
#undef NGRAPH_OP

                template <typename T>
                static void emit_elementwise(EMIT_ARGS)
                {
                    if (out[0].get_size() == 0)
                    {
                        return;
                    }
                    else if (out.size() > 1)
                    {
                        throw std::runtime_error(
                            "Multi-output elementwise ops are not currently supported.");
                    }
                    auto& cuda_emitter =
                        external_function->get_primitive_emitter()->get_cuda_emitter();

                    writer.block_begin();
                    {
                        std::vector<std::string> dtypes;
                        for (auto& arg : args)
                        {
                            dtypes.push_back(arg.get_type());
                        }
                        dtypes.push_back(out[0].get_type());
                        auto ew_index =
                            cuda_emitter->build_elementwise<T>(dtypes, out[0].get_shape());
                        writer << "void* input[] = {" << node_names(args) << "};\n";
                        writer << "void* output[] = {" << node_names(out) << "};\n";
                        writer << "gpu::invoke_primitive(ctx, " << ew_index
                               << ", input, output);\n";
                    }
                    writer.block_end();
                }

                static void emit_ArgReduce(EMIT_ARGS, cudnnReduceTensorOp_t);

                /// \brief Create a list of node names for each arg in args
                /// \param args list of tensor arguments
                /// \param arg_indexes a list of indexes into args for which args to include in
                ///    the output list, so {1, 2} will include args 1 and 2 and skip 0.
                /// \ return returns a string containing "arg0_name, arg1_name, etc."
                static std::string node_names(const std::vector<GPUTensorWrapper>& args,
                                              std::initializer_list<int> arg_indexes = {});
            };

            Shape get_padded_shape(const Shape& input_shape,
                                   const Shape& padding_below,
                                   const Shape& padding_above,
                                   const Shape& padding_interior);
        }
    }
}
