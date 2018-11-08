//*****************************************************************************
// Copyright 2018 Intel Corporation
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
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/subtract.hpp"

#include "ngraph/runtime/cpu/cpu_builder.hpp"
#include "ngraph/runtime/cpu/op/loop_kernel.hpp"

using namespace std;
using namespace ngraph;

#define TI(x) type_index(typeid(x))

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace halide
            {
                static const std::unordered_map<std::type_index,
                                                std::function<Halide::Func(vector<Halide::Func>)>>
                    generators{{TI(ngraph::op::Add),
                                [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = in[0](x) + in[1](x);
                                    return func;
                                }},
                               {TI(ngraph::op::Multiply),
                                [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = in[0](x) * in[1](x);
                                    return func;
                                }},
                               {TI(ngraph::op::Negative),
                                [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = -in[0](x);
                                    return func;
                                }},
                               {TI(ngraph::op::Abs),
                                [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = Halide::abs(in[0](x));
                                    return func;
                                }},
                               // {TI(ngraph::op::Abs), abse},
                               // {TI(ngraph::op::Minimum), mine},
                               // {TI(ngraph::op::Relu), maxe},
                               // {TI(ngraph::op::Maximum), maxe},
                               // {TI(ngraph::op::Add), adde},
                               // {TI(ngraph::op::Negative), nege},
                               // {TI(ngraph::op::Subtract), sube},
                               {TI(ngraph::op::Divide),
                                [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = in[0](x) / in[1](x);
                                    return func;
                                }},
                               {TI(ngraph::op::Maximum),
                                [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = Halide::max(in[0](x), 0);
                                    return func;
                                }},
                               {TI(ngraph::op::Minimum),
                                [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = Halide::min(in[0](x), 0);
                                    return func;
                                }},
                               {TI(ngraph::op::Relu), [](vector<Halide::Func> in) {
                                    Halide::Var x;
                                    Halide::Func func;
                                    func(x) = Halide::max(in[0](x), 0);
                                    return func;
                                }}};
            }

            template <>
            void Builder::BUILDER_DECL(ngraph::runtime::cpu::op::LoopKernel)
            {
                const ngraph::runtime::cpu::op::LoopKernel* hs =
                    static_cast<const ngraph::runtime::cpu::op::LoopKernel*>(node);

                auto& halide_functions = external_function->get_halide_functions();
                auto& subgraph_params = external_function->get_subgraph_params();
                auto& subgraph_param_sizes = external_function->get_subgraph_param_sizes();
                auto& subgraph_param_ptrs = external_function->get_subgraph_param_ptrs();

                std::set<std::string> param_names;
                for (const auto& op : hs->get_node_list())
                {
                    if (!halide::generators.count(TI(*op)))
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
                            if (param_names.count(tensor_name) == 0)
                            {
                                param_names.insert(tensor_name);
                                subgraph_params[tensor_name] =
                                    Halide::ImageParam(Halide::Float(32), 1, tensor_name);
                                subgraph_param_sizes[tensor_name] =
                                    shape_size(input.get_output().get_tensor_ptr()->get_shape());
                                subgraph_param_ptrs.emplace(
                                    tensor_name, external_function->get_tensor_data(tensor_name));
                                inputs.emplace_back(subgraph_params[tensor_name]);
                            }
                            else
                            {
                                inputs.emplace_back(subgraph_params[tensor_name]);
                            }
                        }
                    }
                    //TODO: this needs to be extended to support multi-output ops inside
                    //a subgraph
                    if (op->get_outputs().size() > 1)
                    {
                        throw ngraph_error("no multi-output ops in a LoopKernel");
                    }
                    halide_functions[op->get_output_tensor_ptr()->get_name()] =
                        halide::generators.at(TI(*op))(inputs);
                }

                auto& functors = external_function->get_functors();

                std::vector<std::tuple<void*&, size_t>> buffers_data;
                std::vector<Halide::Expr> results;

                auto output_nodes = hs->get_kernel_outputs();
                Halide::Var x;
                for (size_t i = 0; i < output_nodes.size(); i++)
                {
                    auto result_func =
                        halide_functions[output_nodes.at(i)->get_output_tensor_ptr()->get_name()];
                    results.push_back((result_func(x) + 0));
                    auto& out_tensor = external_function->get_tensor_data(out[i].get_name());
                    buffers_data.push_back(
                        std::tuple<void*&, size_t>(out_tensor, out[i].get_size()));
                }

                Halide::Func terminal_func;
                terminal_func(x) = Halide::Tuple(results);
                auto functor =
                    [&, terminal_func, buffers_data, param_names](CPURuntimeContext* ctx) mutable {

                        std::vector<Halide::Argument> halide_args;
                        for (auto& param : param_names)
                        {
                            Halide::Buffer<float> param_buffer(
                                static_cast<float*>(subgraph_param_ptrs.at(param).get()),
                                subgraph_param_sizes.at(param));
                            subgraph_params[param].set(param_buffer);
                        }
                        std::vector<Halide::Buffer<>> buffers;
                        for (auto tuple : buffers_data)
                        {
                            buffers.push_back(Halide::Buffer<float>(
                                static_cast<float*>(std::get<0>(tuple)), std::get<1>(tuple)));
                        }
                        Halide::Realization r(buffers);
                        terminal_func.realize(r);

                    };
                functors.emplace_back(functor);
            }
        }
    }
}
