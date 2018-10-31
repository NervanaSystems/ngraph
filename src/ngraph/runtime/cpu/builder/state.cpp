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
                function<void(CPURuntimeContext*)> functor;

                auto& arg_tensor = external_function->get_tensor_data(args[0].get_name());
                auto& out_tensor = external_function->get_tensor_data(out[0].get_name());

                size_t element_count = out[0].get_size();

                auto index = external_function->add_state(
                    ngraph::RNGState::create_rng_state(gm->get_seed(), gm->get_probability()));

                if (args[0].get_element_type() == element::f32)
                {
                    functor = [&, index, element_count](CPURuntimeContext* ctx) {
                        bool training = static_cast<bool>(static_cast<float*>(arg_tensor)[0]);
                        reference::generate_mask(static_cast<float*>(out_tensor),
                                                 element_count,
                                                 static_cast<RNGState*>(ctx->states[index]),
                                                 training);
                    };
                }
                else if (args[0].get_element_type() == element::f64)
                {
                    functor = [&, index, element_count](CPURuntimeContext* ctx) {
                        bool training = static_cast<bool>(static_cast<double*>(arg_tensor)[0]);
                        reference::generate_mask(static_cast<double*>(out_tensor),
                                                 element_count,
                                                 static_cast<RNGState*>(ctx->states[index]),
                                                 training);
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

            REGISTER_OP_BUILDER(GenerateMask);
        }
    }
}
