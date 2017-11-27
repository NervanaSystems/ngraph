// ----------------------------------------------------------------------------
// Copyright 2017 Nervana Systems Inc.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// ----------------------------------------------------------------------------

#pragma once

#include <sstream>

#include "ngraph/node.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace interpreter
        {
            template <typename T>
            void op_engine(ngraph::Node& node)
            {
                std::string node_op = node.description();
                if (node_op == "Abs")
                {
                    // node = make_shared<op::Abs>(args[0]);
                }
                else if (node_op == "Acos")
                {
                    // node = make_shared<op::Acos>(args[0]);
                }
                else if (node_op == "Add")
                {
                    // node = make_shared<op::Add>(args[0], args[1]);
                }
                else if (node_op == "Asin")
                {
                    // node = make_shared<op::Asin>(args[0]);
                }
                else if (node_op == "Atan")
                {
                    // node = make_shared<op::Atan>(args[0]);
                }
                else if (node_op == "Broadcast")
                {
                    // auto shape = node_js.at("shape").get<vector<size_t>>();
                    // auto axes = node_js.at("axes").get<set<size_t>>();
                    // node = make_shared<op::Broadcast>(args[0], shape, axes);
                }
                else if (node_op == "Ceiling")
                {
                    // node = make_shared<op::Ceiling>(args[0]);
                }
                else if (node_op == "Concat")
                {
                    // auto axis = node_js.at("axis").get<size_t>();
                    // node = make_shared<op::Concat>(args, axis);
                }
                else if (node_op == "Constant")
                {
                    // auto shape = node_js.at("shape").get<vector<size_t>>();
                    // auto value = node_js.at("value").get<vector<string>>();
                    // node = make_shared<op::Constant>(node_etype, shape, value);
                }
                else if (node_op == "Convert")
                {
                    // auto target_type = node_js.at("target_type").get<element::Type>();
                    // node = make_shared<op::Convert>(args[0], target_type);
                }
                else if (node_op == "Cos")
                {
                    // node = make_shared<op::Cos>(args[0]);
                }
                else if (node_op == "Cosh")
                {
                    // node = make_shared<op::Cosh>(args[0]);
                }
                else if (node_op == "Divide")
                {
                    // node = make_shared<op::Divide>(args[0], args[1]);
                }
                else if (node_op == "Dot")
                {
                    // node = make_shared<op::Dot>(args[0], args[1]);
                }
                else if (node_op == "Equal")
                {
                    // node = make_shared<op::Equal>(args[0], args[1]);
                }
                else if (node_op == "Exp")
                {
                    // node = make_shared<op::Exp>(args[0]);
                }
                else if (node_op == "Floor")
                {
                    // node = make_shared<op::Floor>(args[0]);
                }
                else if (node_op == "FunctionCall")
                {
                    // string function_name = node_js.at("function").get<string>();
                    // shared_ptr<Function> f_ptr = function_map.at(function_name);
                    // node = make_shared<op::FunctionCall>(f_ptr, args);
                }
                // else if (node_op == "GetTupleElement")
                // {
                //     node = make_shared<op::GetTupleElement>(args[0]);
                // }
                else if (node_op == "Greater")
                {
                    // node = make_shared<op::Greater>(args[0], args[1]);
                }
                else if (node_op == "GreaterEq")
                {
                    // node = make_shared<op::GreaterEq>(args[0], args[1]);
                }
                else if (node_op == "Less")
                {
                    // node = make_shared<op::Less>(args[0], args[1]);
                }
                else if (node_op == "LessEq")
                {
                    // node = make_shared<op::LessEq>(args[0], args[1]);
                }
                else if (node_op == "Log")
                {
                    // node = make_shared<op::Log>(args[0]);
                }
                else if (node_op == "Maximum")
                {
                    // node = make_shared<op::Maximum>(args[0], args[1]);
                }
                else if (node_op == "Minimum")
                {
                    // node = make_shared<op::Minimum>(args[0], args[1]);
                }
                else if (node_op == "Multiply")
                {
                    // node = make_shared<op::Multiply>(args[0], args[1]);
                }
                else if (node_op == "Negative")
                {
                    // node = make_shared<op::Negative>(args[0]);
                }
                else if (node_op == "NotEqual")
                {
                    // node = make_shared<op::NotEqual>(args[0], args[1]);
                }
                else if (node_op == "Parameter")
                {
                    NGRAPH_INFO << node_op;
                    // auto shape = node_js.at("shape");
                    // node = make_shared<op::Parameter>(node_etype, shape);
                }
                else if (node_op == "Power")
                {
                    // node = make_shared<op::Power>(args[0], args[1]);
                }
                else if (node_op == "Reduce")
                {
                    // auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
                    // node = make_shared<op::Reduce>(args[0], args[1], function_ptr, reduction_axes);
                }
                else if (node_op == "Remainder")
                {
                    // node = make_shared<op::Remainder>(args[0], args[1]);
                }
                else if (node_op == "Reshape")
                {
                    // auto input_order = node_js.at("input_order").get<vector<size_t>>();
                    // auto output_shape = node_js.at("output_shape").get<vector<size_t>>();
                    // node = make_shared<op::Reshape>(args[0], input_order, output_shape);
                }
                else if (node_op == "Select")
                {
                    // node = make_shared<op::Select>(args[0], args[1], args[2]);
                }
                else if (node_op == "Sign")
                {
                    // node = make_shared<op::Sign>(args[0]);
                }
                else if (node_op == "Sin")
                {
                    // node = make_shared<op::Sin>(args[0]);
                }
                else if (node_op == "Sinh")
                {
                    // node = make_shared<op::Sinh>(args[0]);
                }
                else if (node_op == "Slice")
                {
                    // auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
                    // auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
                    // auto step = node_js.at("step").get<vector<size_t>>();
                    // node = make_shared<op::Slice>(args[0], lower_bounds, upper_bounds, step);
                }
                else if (node_op == "Subtract")
                {
                    // node = make_shared<op::Subtract>(args[0], args[1]);
                }
                else if (node_op == "Sum")
                {
                    // auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
                    // node = make_shared<op::Sum>(args[0], reduction_axes);
                }
                else if (node_op == "Tan")
                {
                    // node = make_shared<op::Tan>(args[0]);
                }
                else if (node_op == "Tanh")
                {
                    // node = make_shared<op::Tanh>(args[0]);
                }
                else if (node_op == "Tuple")
                {
                    // node = make_shared<op::Tuple>(args);
                }
                else
                {
                    std::stringstream ss;
                    ss << "unsupported op " << node_op;
                    throw std::runtime_error(ss.str());
                }
            }
        }
    }
}
