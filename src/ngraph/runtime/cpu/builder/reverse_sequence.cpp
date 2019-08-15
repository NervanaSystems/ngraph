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

#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/kernel/reverse_sequence.hpp"

using namespace std;
using namespace ngraph;

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::op::ReverseSequence)
            {
                auto rev_seq = static_cast<const ngraph::op::ReverseSequence*>(node);

                auto& functors = external_function->get_functors();

                auto arg_buffer_index = external_function->get_buffer_index(args[0].get_name());
                auto seq_len_buffer_index = external_function->get_buffer_index(args[1].get_name());
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());

                auto arg_shape = args[0].get_shape();

                auto sequence_axis = rev_seq->get_sequence_axis();
                auto batch_axis = rev_seq->get_batch_axis();

                std::function<decltype(runtime::cpu::kernel::reverse_sequence<int, int, 4>)> kernel;

                if (args[1].get_element_type() == element::i32)
                {
                    SELECT_KERNEL_BY_RANK(kernel,
                                          args[0].get_element_type(),
                                          arg_shape.size(),
                                          runtime::cpu::kernel::reverse_sequence_sli32);
                }
                else
                {
                    throw ngraph_error("Unsupported sequence length type " +
                                       args[1].get_element_type().c_type_string() +
                                       " requires a kernel instantiation to handle this type");
                }

                auto functor = [&,
                                kernel,
                                arg_shape,
                                batch_axis,
                                sequence_axis,
                                arg_buffer_index,
                                seq_len_buffer_index,
                                out_buffer_index](CPURuntimeContext* ctx,
                                                  CPUExecutionContext* ectx) {
                    kernel(ctx->buffer_data[arg_buffer_index],
                           ctx->buffer_data[out_buffer_index],
                           arg_shape,
                           batch_axis,
                           sequence_axis,
                           ctx->buffer_data[seq_len_buffer_index],
                           ectx->arena);
                };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(ReverseSequence);
#ifdef NGRAPH_CPU_STATIC_LIB_ENABLE
            void register_builders_reverse_sequence_cpp() {}
#endif
        }
    }
}
