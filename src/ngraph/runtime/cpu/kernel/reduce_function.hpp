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

#define EIGEN_USE_THREADS
#include <unsupported/Eigen/CXX11/Tensor>

#include "ngraph/axis_set.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/runtime/cpu/cpu_external_function.hpp"
#include "ngraph/runtime/cpu/cpu_tensor_view.hpp"
#include "ngraph/runtime/cpu/kernel/eigen_thread_pool.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace kernel
            {
                template <typename ElementType>
                struct Reducer
                {
                    static const bool PacketAccess = false;
                    static const bool IsStateful = false;

                    ElementType initial;
                    const std::shared_ptr<CPU_ExternalFunction>& external_function;
                    std::shared_ptr<Backend> backend;

                    Reducer(ElementType x, const std::shared_ptr<CPU_ExternalFunction>& ef)
                        : initial(x)
                        , external_function(ef)
                        , backend(runtime::Backend::create("CPU"))
                    {
                    }

                    void reduce(const ElementType v, ElementType* R)
                    {
                        TensorViewPtrs inputs, outputs;
                        ElementType p __attribute__((aligned(NGRAPH_CPU_ALIGNMENT))) = v;
                        ElementType q __attribute__((aligned(NGRAPH_CPU_ALIGNMENT))) = *R;
                        ElementType r __attribute__((aligned(NGRAPH_CPU_ALIGNMENT)));

                        inputs.emplace_back(backend->create_tensor(
                            ngraph::element::from<ElementType>(), Shape{}, &p));
                        inputs.emplace_back(backend->create_tensor(
                            ngraph::element::from<ElementType>(), Shape{}, &q));
                        outputs.emplace_back(backend->create_tensor(
                            ngraph::element::from<ElementType>(), Shape{}, &r));
                        auto call_frame = external_function->make_call_frame();
                        call_frame->call(outputs, inputs);
                        *R = r;
                    }
                    ElementType initialize() const { return initial; }
                    ElementType finalize(const ElementType R) const { return R; }
                };

                template <typename ElementType, unsigned int Rank, unsigned int ReductionDims>
                void reduce_function(void* input0,
                                     void* input1,
                                     void* output,
                                     const Shape& input_shape,
                                     const Shape& output_shape,
                                     const AxisSet& reduction_axes,
                                     const std::shared_ptr<CPU_ExternalFunction>& external_function)
                {
                    Eigen::array<Eigen::Index, Rank> in_dims;
                    Eigen::array<Eigen::Index, Rank - ReductionDims> out_dims;
                    Eigen::array<Eigen::Index, ReductionDims> reduction_dims;

                    for (int i = 0; i < Rank; i++)
                    {
                        in_dims[i] = input_shape[i];
                    }

                    for (int i = 0; i < Rank - ReductionDims; i++)
                    {
                        out_dims[i] = output_shape[i];
                    }

                    int i = 0;
                    for (auto axis : reduction_axes)
                    {
                        reduction_dims[i++] = axis;
                    }

                    Eigen::TensorMap<
                        Eigen::Tensor<ElementType, Rank - ReductionDims, Eigen::RowMajor>>
                        out(static_cast<ElementType*>(output), out_dims);
                    Eigen::TensorMap<Eigen::Tensor<ElementType, Rank, Eigen::RowMajor>> in(
                        static_cast<ElementType*>(input0), in_dims);
                    Reducer<ElementType> reducer(*static_cast<ElementType*>(input1),
                                                 external_function);
                    out.device(eigen::global_thread_pool_device) =
                        in.reduce(reduction_dims, reducer);
                }

                template <typename ElementType, unsigned int Rank>
                void reduce_function_1rd(
                    void* input0,
                    void* input1,
                    void* output,
                    const Shape& input_shape,
                    const Shape& output_shape,
                    const AxisSet& reduction_axes,
                    const std::shared_ptr<CPU_ExternalFunction>& external_function)
                {
                    reduce_function<ElementType, Rank, 1>(input0,
                                                          input1,
                                                          output,
                                                          input_shape,
                                                          output_shape,
                                                          reduction_axes,
                                                          external_function);
                }

                template <typename ElementType>
                void reduce_function_2d_2rd(
                    void* input0,
                    void* input1,
                    void* output,
                    const Shape& input_shape,
                    const Shape& output_shape,
                    const AxisSet& reduction_axes,
                    const std::shared_ptr<CPU_ExternalFunction>& external_function)
                {
                    reduce_function<ElementType, 2, 2>(input0,
                                                       input1,
                                                       output,
                                                       input_shape,
                                                       output_shape,
                                                       reduction_axes,
                                                       external_function);
                }

                template <typename ElementType>
                void reduce_function_3d_2rd(
                    void* input0,
                    void* input1,
                    void* output,
                    const Shape& input_shape,
                    const Shape& output_shape,
                    const AxisSet& reduction_axes,
                    const std::shared_ptr<CPU_ExternalFunction>& external_function)
                {
                    reduce_function<ElementType, 3, 2>(input0,
                                                       input1,
                                                       output,
                                                       input_shape,
                                                       output_shape,
                                                       reduction_axes,
                                                       external_function);
                }
            }
        }
    }
}
