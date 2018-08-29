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

#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/select_and_scatter.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::SelectAndScatter)
            {
                auto select_and_scatter = static_cast<const ngraph::op::SelectAndScatter*>(node);
                auto select_function = select_and_scatter->get_functions()[0];
                auto scatter_function = select_and_scatter->get_functions()[1];

                auto backend = runtime::Backend::create("CPU");

                auto& functors = external_function->get_functors();
                auto& callees = external_function->get_callees();

                // Note: We bypass the completely broken ngraph "backend" API here
                auto element_type = node->get_output_element_type(0);

                if (element_type != element::f32)
                {
                    throw ngraph_error(
                        "CPU direct execution mode does not support non-float inputs, use compiled "
                        "mode instead");
                }

                auto arg0_shape = args[0].get_shape();
                auto& arg0_tensor = external_function->get_tensor_data(args[0].get_name());
                auto arg1_shape = args[1].get_shape();
                auto& arg1_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& arg2_tensor = external_function->get_tensor_data(args[2].get_name());

                auto out_shape = out[0].get_shape();
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                auto window_shape = select_and_scatter->get_window_shape();
                auto window_movement_strides = select_and_scatter->get_window_movement_strides();

                if (!callees.count(select_function->get_name()))
                {
                    callees[select_function->get_name()] =
                        make_shared<CPU_ExternalFunction>(select_function);
                }
                if (!callees.count(scatter_function->get_name()))
                {
                    callees[scatter_function->get_name()] =
                        make_shared<CPU_ExternalFunction>(scatter_function);
                }

                auto& select_external_function = callees[select_function->get_name()];
                auto& scatter_external_function = callees[scatter_function->get_name()];

                auto select = [&, backend](float x, float y) {
                    TensorViewPtrs inputs, outputs;
                    char output;
                    inputs.emplace_back(backend->create_tensor(element::f32, Shape{}, &x));
                    inputs.emplace_back(backend->create_tensor(element::f32, Shape{}, &y));
                    outputs.emplace_back(backend->create_tensor(element::f32, Shape{}, &output));
                    select_external_function->make_call_frame()->call(outputs, inputs);
                    return output;
                };

                auto scatter = [&, backend](float x, float y) {
                    TensorViewPtrs inputs, outputs;
                    float output;
                    inputs.emplace_back(backend->create_tensor(element::f32, Shape{}, &x));
                    inputs.emplace_back(backend->create_tensor(element::f32, Shape{}, &y));
                    outputs.emplace_back(backend->create_tensor(element::f32, Shape{}, &output));
                    scatter_external_function->make_call_frame()->call(outputs, inputs);
                    return output;
                };

                auto functor = [&,
                                backend,
                                select,
                                scatter,
                                arg0_shape,
                                arg1_shape,
                                out_shape,
                                window_shape,
                                window_movement_strides](CPURuntimeContext* ctx) {
                    reference::select_and_scatter<float>(static_cast<float*>(arg0_tensor),
                                                         static_cast<float*>(arg1_tensor),
                                                         static_cast<float*>(arg2_tensor),
                                                         static_cast<float*>(out_tensor),
                                                         arg0_shape,
                                                         arg1_shape,
                                                         out_shape,
                                                         select,
                                                         scatter,
                                                         window_shape,
                                                         window_movement_strides);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(SelectAndScatter);
        }
    }
}
