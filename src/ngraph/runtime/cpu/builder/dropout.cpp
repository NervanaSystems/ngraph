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

#include "ngraph/runtime/cpu/op/dropout.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/dropout.hpp"
#include "ngraph/state/rng_state.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::Dropout)
            {
                auto& functors = external_function->get_functors();

                auto drop = static_cast<const ngraph::op::Dropout*>(node);
                CPUKernelFunctor functor;

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out0_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto out1_buffer_index = external_function->get_buffer_index(out[1].get_name());

                auto in_shape = args[0].get_shape();
                size_t element_count = out[0].get_size();

                unsigned int seed = static_cast<unsigned int> (drop->get_seed());
                double value = drop->get_value();
                auto index = external_function->add_state(
                                ngraph::RNGState::create_rng_state(seed, value));

                if (args[0].get_element_type() == element::f32)
                {
                    functor = [&, index, element_count, arg_buffer_index, out0_buffer_index,
                                 out1_buffer_index, in_shape, value]
                                 (CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        bool training = true; //TODO:
                        /*bool training = static_cast<bool>( 
                            static_cast<float*>(ctx->buffer_data[arg_buffer_index])[0]);*/

                        runtime::cpu::kernel::generate_dropout(
                            static_cast<float*>(ctx->buffer_data[arg_buffer_index]),
                            static_cast<float*>(ctx->buffer_data[out0_buffer_index]),
                            static_cast<float*>(ctx->buffer_data[out1_buffer_index]),
                            element_count,
                            in_shape,
                            static_cast<RNGState*>(ctx->states[index]),
                            training,
                            value);
                    };
                }
                else if (args[0].get_element_type() == element::f64)
                {
                    functor = [&, index, element_count, arg_buffer_index, out0_buffer_index,
                                out1_buffer_index, in_shape, value]
                                (CPURuntimeContext* ctx, CPUExecutionContext* ectx) {
                        bool training = true;
                        /*bool training = static_cast<bool>(
                            static_cast<double*>(ctx->buffer_data[arg_buffer_index])[0]);*/
                        runtime::cpu::kernel::generate_dropout(
                            static_cast<double*>(ctx->buffer_data[arg_buffer_index]),
                            static_cast<double*>(ctx->buffer_data[out0_buffer_index]),
                            static_cast<double*>(ctx->buffer_data[out1_buffer_index]),
                            element_count,
                            in_shape,
                            static_cast<RNGState*>(ctx->states[index]),
                            training,
                            value);
                    };
                }
                else
                {
                    throw ngraph_error(std::string("Unsupported type") +
                                       args[0].get_element_type().c_type_string() +
                                       "for Dropout");
                }
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(Dropout);
        }
    }
}
