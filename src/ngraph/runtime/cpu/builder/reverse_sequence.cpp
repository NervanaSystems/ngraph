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

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& seq_len_tensor = external_function->get_tensor_data(args[1].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

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

                auto functor =
                    [&, kernel, arg_shape, batch_axis, sequence_axis](CPURuntimeContext* ctx) {
                        kernel(arg_tensor,
                               out_tensor,
                               arg_shape,
                               batch_axis,
                               sequence_axis,
                               seq_len_tensor);
                    };
                functors.emplace_back(functor);
            }

            REGISTER_OP_BUILDER(ReverseSequence);
        }
    }
}
