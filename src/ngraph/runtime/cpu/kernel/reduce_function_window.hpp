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

#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"

#include "ngraph/runtime/reference/reduce_window.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType>
                void reduce_function_window(
                    void* input0,
                    void* input1,
                    void* output,
                    const Shape& input_shape,
                    const Shape& output_shape,
                    const Shape& window_shape,
                    const Strides& window_movement_strides,
                    const std::shared_ptr<CPU_ExternalFunction>& external_function)
                {
                    auto backend = runtime::Backend::create("CPU");

                    auto reducer = [&](ElementType a, ElementType b) {
                        TensorViewPtrs inputs, outputs;

                        ElementType p __attribute__((aligned(NGRAPH_CPU_ALIGNMENT))) = a;
                        ElementType q __attribute__((aligned(NGRAPH_CPU_ALIGNMENT))) = b;
                        ElementType r __attribute__((aligned(NGRAPH_CPU_ALIGNMENT)));

                        inputs.emplace_back(backend->create_tensor(
                            ngraph::element::from<ElementType>(), Shape{}, &p));
                        inputs.emplace_back(backend->create_tensor(
                            ngraph::element::from<ElementType>(), Shape{}, &q));
                        outputs.emplace_back(backend->create_tensor(
                            ngraph::element::from<ElementType>(), Shape{}, &r));

                        auto call_frame = external_function->make_call_frame();
                        call_frame->call(outputs, inputs);

                        return r;
                    };

                    reference::reduce_window<ElementType>(static_cast<const ElementType*>(input0),
                                                          static_cast<const ElementType*>(input1),
                                                          static_cast<ElementType*>(output),
                                                          input_shape,
                                                          output_shape,
                                                          reducer,
                                                          window_shape,
                                                          window_movement_strides);
                }
            }
        }
    }
}
