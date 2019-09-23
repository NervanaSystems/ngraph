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

#include <Halide.h>
#include <HalideBuffer.h>
#include <functional>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/op/add.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/relu.hpp"

#include "halide_generators.hpp"
#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/op/halide_op.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            template <>
            void Builder::BUILDER_DECL(ngraph::runtime::cpu::op::HalideOp)
            {
                const ngraph::runtime::cpu::op::HalideOp* hs =
                    static_cast<const ngraph::runtime::cpu::op::HalideOp*>(node);

                const auto& generators = ngraph::runtime::cpu::halide::get_halide_generators();

                auto& halide_functions = external_function->get_halide_functions();
                auto& subgraph_params = external_function->get_subgraph_params();
                auto& subgraph_param_sizes = external_function->get_subgraph_param_sizes();
                auto& subgraph_param_indices = external_function->get_subgraph_param_indices();

                for (const auto& op : hs->get_ops())
                {
                    if (!generators.count(TI(*op)))
                    {
                        throw ngraph_error("Invalid op in halide subgraph");
                    }
                    vector<Halide::Func> inputs;
                    for (const auto& input : op->get_inputs())
                    {
                        auto tensor_name = input.get_output().get_tensor_ptr()->get_name();
                        if (halide_functions.count(tensor_name))
                        {
                            inputs.emplace_back(halide_functions[tensor_name]);
                        }
                        else
                        {
                            subgraph_params[tensor_name] = Halide::ImageParam(Halide::Float(32), 1);
                            subgraph_param_sizes[tensor_name] =
                                shape_size(input.get_output().get_tensor_ptr()->get_shape());
                            subgraph_param_indices.emplace(
                                tensor_name, external_function->get_buffer_index(tensor_name));
                            inputs.emplace_back(subgraph_params[tensor_name]);
                        }
                    }
                    halide_functions[op->get_output_tensor_ptr()->get_name()] =
                        generators.at(TI(*op))(inputs);
                }

                auto out_tensor_name = hs->get_ops().back()->get_output_tensor_ptr()->get_name();
                auto& functors = external_function->get_functors();
                auto out_buffer_index = external_function->get_buffer_index(out[0].get_name());
                auto& terminal_func = halide_functions[out_tensor_name];
                auto out_size = out[0].get_size();

                auto functor = [&, out_size, out_buffer_index](CPURuntimeContext* ctx,
                                                               CPUExecutionContext* ectx) {
                    for (auto& param : subgraph_params)
                    {
                        Halide::Buffer<float> param_buffer(
                            static_cast<float*>(
                                ctx->buffer_data[subgraph_param_indices.at(param.first)]),
                            subgraph_param_sizes.at(param.first));
                        param.second.set(param_buffer);
                    }
                    Halide::Buffer<float> out_buffer(
                        static_cast<float*>(ctx->buffer_data[out_buffer_index]), out_size);
                    terminal_func.realize(out_buffer);
                };
                functors.emplace_back(functor);
            }
        }
    }
}
