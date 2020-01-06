//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/op/experimental/random_uniform.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/reference/random_uniform.hpp"
#include "ngraph/state/uniform_rng_state.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <typename T>
            CPUKernelFunctor prepare_functor(const Node* node,
                                             const vector<TensorViewWrapper>& args,
                                             const vector<TensorViewWrapper>& out,
                                             CPU_ExternalFunction* external_function)
            {
                auto ru = static_cast<const ngraph::op::RandomUniform*>(node);
                CPUKernelFunctor functor;

                auto arg0_buffer_index =
                    external_function->get_buffer_index(args[0].get_name()); // min_val
                auto arg1_buffer_index =
                    external_function->get_buffer_index(args[1].get_name()); // max_val
                auto arg3_buffer_index =
                    external_function->get_buffer_index(args[3].get_name()); // use_fixed_seed

                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                size_t element_count = out[0].get_size();

                auto index = external_function->add_state(new ngraph::UniformRNGState());
                auto fixed_seed = ru->get_fixed_seed();

                functor = [&,
                           index,
                           element_count,
                           arg0_buffer_index,
                           arg1_buffer_index,
                           arg3_buffer_index,
                           out_buffer_index,
                           fixed_seed](CPURuntimeContext* ctx, CPUExecutionContext* /* ectx */) {
                    // TODO: get shape when required

                    T min_val = static_cast<T*>(ctx->buffer_data[arg0_buffer_index])[0];
                    T max_val = static_cast<T*>(ctx->buffer_data[arg1_buffer_index])[0];
                    bool use_fixed_seed = static_cast<bool>(
                        static_cast<char*>(ctx->buffer_data[arg3_buffer_index])[0]);

                    if (!use_fixed_seed)
                    {
                        reference::random_uniform<T>(
                            static_cast<T*>(ctx->buffer_data[out_buffer_index]),
                            min_val,
                            max_val,
                            element_count,
                            static_cast<UniformRNGState*>(ctx->states[index]));
                    }
                    else
                    {
                        reference::random_uniform_with_fixed_seed<T>(
                            static_cast<T*>(ctx->buffer_data[out_buffer_index]),
                            min_val,
                            max_val,
                            element_count,
                            fixed_seed);
                    }
                };
                return functor;
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::op::RandomUniform)
            {
                auto& functors = external_function->get_functors();
                CPUKernelFunctor functor;
                if (args[2].get_element_type() != element::i64)
                {
                    throw ngraph_error("Unsupported index 2 element type");
                }
                auto element_type = args[0].get_element_type();
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
#endif
                switch (element_type)
                {
                case element::Type_t::undefined:
                    NGRAPH_CHECK(false,
                                 "Encountered 'undefined' element type in fold_constant_convert");
                    break;
                case element::Type_t::dynamic:
                    NGRAPH_CHECK(false,
                                 "Encountered 'dynamic' element type in fold_constant_convert");
                    break;
                case element::Type_t::u1:
                    NGRAPH_CHECK(false, "Encountered 'u1' element type in fold_constant_convert");
                    break;
                case element::Type_t::boolean:
                    functor = prepare_functor<char>(node, args, out, external_function);
                    break;
                case element::Type_t::bf16:
                    functor = prepare_functor<bfloat16>(node, args, out, external_function);
                    break;
                case element::Type_t::f16:
                    functor = prepare_functor<float16>(node, args, out, external_function);
                    break;
                case element::Type_t::f32:
                    functor = prepare_functor<float>(node, args, out, external_function);
                    break;
                case element::Type_t::f64:
                    functor = prepare_functor<double>(node, args, out, external_function);
                    break;
                case element::Type_t::i8:
                    functor = prepare_functor<int8_t>(node, args, out, external_function);
                    break;
                case element::Type_t::i16:
                    functor = prepare_functor<int16_t>(node, args, out, external_function);
                    break;
                case element::Type_t::i32:
                    functor = prepare_functor<int32_t>(node, args, out, external_function);
                    break;
                case element::Type_t::i64:
                    functor = prepare_functor<int64_t>(node, args, out, external_function);
                    break;
                case element::Type_t::u8:
                    functor = prepare_functor<uint8_t>(node, args, out, external_function);
                    break;
                case element::Type_t::u16:
                    functor = prepare_functor<uint16_t>(node, args, out, external_function);
                    break;
                case element::Type_t::u32:
                    functor = prepare_functor<uint32_t>(node, args, out, external_function);
                    break;
                case element::Type_t::u64:
                    functor = prepare_functor<uint64_t>(node, args, out, external_function);
                    break;
                    NGRAPH_UNREACHABLE("Unexpected switch case");
                }

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

                functors.emplace_back(functor);
            }

            void register_builders_random_uniform_cpp() { REGISTER_OP_BUILDER(RandomUniform); }
        }
    }
}
