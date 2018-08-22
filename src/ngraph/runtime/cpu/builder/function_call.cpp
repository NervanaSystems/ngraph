/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include "ngraph/op/function_call.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/tensor_view.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::FunctionCall)
            {
                auto function_call = static_cast<const ngraph::op::FunctionCall*>(node);
                auto function = function_call->get_functions()[0];
                auto backend = runtime::Backend::create("CPU");

                auto& functors = external_function->get_functors();
                auto& callees = external_function->get_callees();

                // Note: We bypass the completely broken ngraph "backend" API here
                vector<reference_wrapper<void *>> arg_tensors, out_tensors;
                vector<Shape> arg_shapes, out_shapes;
                vector<element::Type> arg_types, out_types;

                for (const auto& arg : args)
                {
                    arg_shapes.emplace_back(arg.get_shape());
                    arg_types.emplace_back(arg.get_element_type());
                    arg_tensors.emplace_back(external_function->get_tensor_data(arg.get_name()));
                }

                for (const auto& result : out)
                {
                    out_shapes.emplace_back(result.get_shape());
                    out_types.emplace_back(result.get_element_type());
                    out_tensors.emplace_back(external_function->get_tensor_data(result.get_name()));
                }

                if (!callees.count(function->get_name()))
                {
                    callees[function->get_name()] = make_shared<CPU_ExternalFunction>(function);
                }

                auto& callee_external_function = callees[function->get_name()];

                auto functor = [&,
                                backend,
                                arg_shapes,
                                arg_types,
                                arg_tensors,
                                out_shapes,
                                out_types,
                                out_tensors](CPURuntimeContext* ctx) {
                    TensorViewPtrs inputs, outputs;
                    for (int i = 0; i < arg_shapes.size(); i++)
                    {
                        inputs.emplace_back(
                            backend->create_tensor(arg_types[i], arg_shapes[i], arg_tensors[i]));
                    }
                    for (int i = 0; i < out_shapes.size(); i++)
                    {
                        outputs.emplace_back(
                            backend->create_tensor(out_types[i], out_shapes[i], out_tensors[i]));
                    }

                    auto call_frame = callee_external_function->make_call_frame();
                    call_frame->call(outputs, inputs);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(FunctionCall);
        }
    }
}
