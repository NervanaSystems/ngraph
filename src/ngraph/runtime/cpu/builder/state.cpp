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

#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/generate_mask.hpp"
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
            void Builder::BUILDER_DECL(ngraph::op::GenerateMask)
            {
                auto& functors = external_function->get_functors();

                auto gm = static_cast<const ngraph::op::GenerateMask*>(node);
                CPUKernelFunctor functor;

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                size_t element_count = out[0].get_size();

                auto arg2_buffer_index =
                    external_function->get_buffer_index(args[2].get_name()); // use_seed
                auto arg3_buffer_index =
                    external_function->get_buffer_index(args[3].get_name()); // seed
                auto arg4_buffer_index =
                    external_function->get_buffer_index(args[4].get_name()); // prob

                auto seed_attr = gm->get_use_seed() ? gm->get_seed() : 0;
                auto index = external_function->add_state(
                    ngraph::RNGState::create_rng_state(seed_attr, gm->get_probability()));

                if (args[0].get_element_type() == element::f32)
                {
                    functor = [&,
                               index,
                               element_count,
                               arg_buffer_index,
                               out_buffer_index,
                               arg2_buffer_index,
                               arg3_buffer_index,
                               arg4_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                        bool training = static_cast<bool>(
                            static_cast<float*>(ctx->buffer_data[arg_buffer_index])[0]);
                        // TODO: get shape when required
                        bool use_seed = static_cast<bool>(
                            static_cast<int32_t*>(ctx->buffer_data[arg2_buffer_index])[0]);
                        uint64_t seed =
                            static_cast<uint64_t*>(ctx->buffer_data[arg3_buffer_index])[0];
                        double prob = static_cast<double*>(ctx->buffer_data[arg4_buffer_index])[0];

                        if (use_seed == false)
                        {
                            reference::generate_mask(
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                static_cast<RNGState*>(ctx->states[index]),
                                training);
                        }
                        else
                        {
                            reference::generate_mask_no_state(
                                static_cast<float*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                training,
                                seed,
                                prob);
                        }
                    };
                }
                else if (args[0].get_element_type() == element::f64)
                {
                    functor = [&,
                               index,
                               element_count,
                               arg_buffer_index,
                               out_buffer_index,
                               arg2_buffer_index,
                               arg3_buffer_index,
                               arg4_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                        bool training = static_cast<bool>(
                            static_cast<double*>(ctx->buffer_data[arg_buffer_index])[0]);
                        // TODO: get shape when required
                        bool use_seed = static_cast<bool>(
                            static_cast<int32_t*>(ctx->buffer_data[arg2_buffer_index])[0]);
                        uint64_t seed =
                            static_cast<uint64_t*>(ctx->buffer_data[arg3_buffer_index])[0];
                        double prob = static_cast<double*>(ctx->buffer_data[arg4_buffer_index])[0];

                        if (use_seed == false)
                        {
                            reference::generate_mask(
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                static_cast<RNGState*>(ctx->states[index]),
                                training);
                        }
                        else
                        {
                            reference::generate_mask_no_state(
                                static_cast<double*>(ctx->buffer_data[out_buffer_index]),
                                element_count,
                                training,
                                seed,
                                prob);
                        }
                    };
                }
                else
                {
                    throw ngraph_error(std::string("Unsupported type") +
                                       args[0].get_element_type().c_type_string() +
                                       "for GenerateMask");
                }
                functors.emplace_back(functor);
            }

            void register_builders_state_cpp() { REGISTER_OP_BUILDER(GenerateMask); }
        }
    }
}
