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

#include "halide_generators.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"

using namespace ngraph;
using namespace std;

#define TI(x) std::type_index(typeid(x))

namespace ngraph
{
    namespace runtime
    {
        namespace cpu
        {
            namespace halide
            {
                const std::unordered_map<std::type_index,
                                         std::function<Halide::Func(std::vector<Halide::Func>)>>&
                    get_halide_generators()
                {
                    const static std::unordered_map<
                        std::type_index,
                        std::function<Halide::Func(std::vector<Halide::Func>)>>
                        generators{{TI(ngraph::op::Add),
                                    [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = in[0](x) + in[1](x);
                                        return func;
                                    }},
                                   {TI(ngraph::op::Multiply),
                                    [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = in[0](x) * in[1](x);
                                        return func;
                                    }},
                                   {TI(ngraph::op::Negative),
                                    [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = -in[0](x);
                                        return func;
                                    }},
                                   {TI(ngraph::op::Abs),
                                    [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = Halide::abs(in[0](x));
                                        return func;
                                    }},
                                   {TI(ngraph::op::Divide),
                                    [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = in[0](x) / in[1](x);
                                        return func;
                                    }},
                                   {TI(ngraph::op::Maximum),
                                    [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = Halide::max(in[0](x), 0);
                                        return func;
                                    }},
                                   {TI(ngraph::op::Minimum),
                                    [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = Halide::min(in[0](x), 0);
                                        return func;
                                    }},
                                   {TI(ngraph::op::Relu), [](std::vector<Halide::Func> in) {
                                        Halide::Var x;
                                        Halide::Func func;
                                        func(x) = Halide::max(in[0](x), 0);
                                        return func;
                                    }}};

                    return generators;
                }
            }
        }
    }
}
