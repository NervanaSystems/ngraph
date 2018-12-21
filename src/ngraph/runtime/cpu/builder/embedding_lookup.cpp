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

#include <cstring>

#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/embedding_lookup.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::EmbeddingLookup)
            {
                auto& functors = external_function->get_functors();
                auto& tensor_data = external_function->get_tensor_data();

                CPUKernelFunctor functor;

                auto& arg0_tensor = tensor_data[args[0].get_name()];
                auto& arg1_tensor = tensor_data[args[1].get_name()];
                auto& out_tensor = tensor_data[out[0].get_name()];
                if (out[0].get_element_type() != element::f32 &&
                    out[0].get_element_type() != element::f64)
                {
                    throw ngraph_error("Unsupported output element type");
                }
                auto in_shape = args[1].get_shape();
                size_t element_count = shape_size(args[0].get_shape());
                auto out_shape = out[0].get_shape();
                auto element_type = out[0].get_element_type();
                auto index_element_type = args[0].get_element_type();
                if (element_type == element::f32)
                {
                    if (index_element_type == element::f32)
                    {
                        functor = [&, in_shape, element_count](CPURuntimeContext* ctx,
                                                               CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<float, float>(
                                static_cast<float*>(arg0_tensor),
                                static_cast<float*>(arg1_tensor),
                                static_cast<float*>(out_tensor),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i32)
                    {
                        functor = [&, in_shape, element_count](CPURuntimeContext* ctx,
                                                               CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<float, int>(
                                static_cast<int*>(arg0_tensor),
                                static_cast<float*>(arg1_tensor),
                                static_cast<float*>(out_tensor),
                                element_count,
                                in_shape);
                        };
                    }
                    else
                    {
                        throw ngraph_error(
                            "Unsupported index type in CPU Builder for EmbeddingLookup");
                    }
                }
                else if (element_type == element::i32)
                {
                    if (index_element_type == element::f32)
                    {
                        functor = [&, in_shape, element_count](CPURuntimeContext* ctx,
                                                               CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<int, float>(
                                static_cast<float*>(arg0_tensor),
                                static_cast<int*>(arg1_tensor),
                                static_cast<int*>(out_tensor),
                                element_count,
                                in_shape);
                        };
                    }
                    else if (index_element_type == element::i32)
                    {
                        functor = [&, in_shape, element_count](CPURuntimeContext* ctx,
                                                               CPUExecutionContext* ectx) {

                            ngraph::runtime::reference::embedding<int, int>(
                                static_cast<int*>(arg0_tensor),
                                static_cast<int*>(arg1_tensor),
                                static_cast<int*>(out_tensor),
                                element_count,
                                in_shape);
                        };
                    }
                    else
                    {
                        throw ngraph_error(
                            "Unsupported index type in CPU Builder for EmbeddingLookup");
                    }
                }
                else
                {
                    throw ngraph_error("Unsupported type in CPU Builder for ArgMin");
                }

                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(EmbeddingLookup);
        }
    }
}
