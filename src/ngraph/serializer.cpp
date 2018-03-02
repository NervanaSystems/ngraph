/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "ngraph/serializer.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/ops/abs.hpp"
#include "ngraph/ops/acos.hpp"
#include "ngraph/ops/add.hpp"
#include "ngraph/ops/allreduce.hpp"
#include "ngraph/ops/asin.hpp"
#include "ngraph/ops/atan.hpp"
#include "ngraph/ops/avg_pool.hpp"
#include "ngraph/ops/batch_norm.hpp"
#include "ngraph/ops/broadcast.hpp"
#include "ngraph/ops/ceiling.hpp"
#include "ngraph/ops/concat.hpp"
#include "ngraph/ops/constant.hpp"
#include "ngraph/ops/convert.hpp"
#include "ngraph/ops/convolution.hpp"
#include "ngraph/ops/cos.hpp"
#include "ngraph/ops/cosh.hpp"
#include "ngraph/ops/divide.hpp"
#include "ngraph/ops/dot.hpp"
#include "ngraph/ops/equal.hpp"
#include "ngraph/ops/exp.hpp"
#include "ngraph/ops/floor.hpp"
#include "ngraph/ops/function_call.hpp"
#include "ngraph/ops/get_output_element.hpp"
#include "ngraph/ops/greater.hpp"
#include "ngraph/ops/greater_eq.hpp"
#include "ngraph/ops/less.hpp"
#include "ngraph/ops/less_eq.hpp"
#include "ngraph/ops/log.hpp"
#include "ngraph/ops/max.hpp"
#include "ngraph/ops/max_pool.hpp"
#include "ngraph/ops/maximum.hpp"
#include "ngraph/ops/min.hpp"
#include "ngraph/ops/minimum.hpp"
#include "ngraph/ops/multiply.hpp"
#include "ngraph/ops/negative.hpp"
#include "ngraph/ops/not.hpp"
#include "ngraph/ops/not_equal.hpp"
#include "ngraph/ops/one_hot.hpp"
#include "ngraph/ops/pad.hpp"
#include "ngraph/ops/power.hpp"
#include "ngraph/ops/product.hpp"
#include "ngraph/ops/reduce.hpp"
#include "ngraph/ops/reduce_window.hpp"
#include "ngraph/ops/relu.hpp"
#include "ngraph/ops/remainder.hpp"
#include "ngraph/ops/replace_slice.hpp"
#include "ngraph/ops/reshape.hpp"
#include "ngraph/ops/reverse.hpp"
#include "ngraph/ops/select.hpp"
#include "ngraph/ops/select_and_scatter.hpp"
#include "ngraph/ops/sign.hpp"
#include "ngraph/ops/sin.hpp"
#include "ngraph/ops/sinh.hpp"
#include "ngraph/ops/slice.hpp"
#include "ngraph/ops/sqrt.hpp"
#include "ngraph/ops/subtract.hpp"
#include "ngraph/ops/sum.hpp"
#include "ngraph/ops/tan.hpp"
#include "ngraph/ops/tanh.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"

using namespace ngraph;
using namespace std;
using json = nlohmann::json;

template <typename T>
T get_or_default(nlohmann::json& j, const std::string& key, const T& default_value)
{
    T rc;
    try
    {
        rc = j.at(key).get<T>();
    }
    catch (...)
    {
        rc = default_value;
    }
    return rc;
}

static std::shared_ptr<ngraph::Function>
    read_function(const json&, std::unordered_map<std::string, std::shared_ptr<Function>>&);

static json write(const ngraph::Function&);
static json write(const ngraph::Node&);

static json write_element_type(const ngraph::element::Type& n)
{
    json j;
    j = n.c_type_string();
    return j;
}

static element::Type read_element_type(const json& j)
{
    size_t bitwidth = 0;
    bool is_real = false;
    bool is_signed = false;
    string c_type_string = "";
    if (j.is_object())
    {
        bitwidth = j.at("bitwidth").get<size_t>();
        is_real = j.at("is_real").get<bool>();
        is_signed = j.at("is_signed").get<bool>();
        c_type_string = j.at("c_type_string").get<string>();
    }
    else
    {
        string c_type = j.get<string>();
        for (const element::Type* t : element::Type::get_known_types())
        {
            if (t->c_type_string() == c_type)
            {
                bitwidth = t->bitwidth();
                is_real = t->is_real();
                is_signed = t->is_signed();
                c_type_string = t->c_type_string();
                break;
            }
        }
    }
    return element::Type(bitwidth, is_real, is_signed, c_type_string);
}

string ngraph::serialize(shared_ptr<ngraph::Function> func, size_t indent)
{
    json j;
    vector<json> functions;
    traverse_functions(func,
                       [&](shared_ptr<ngraph::Function> f) { functions.push_back(write(*f)); });
    for (auto it = functions.rbegin(); it != functions.rend(); it++)
    {
        j.push_back(*it);
    }

    string rc;
    if (indent == 0)
    {
        rc = j.dump();
    }
    else
    {
        rc = j.dump(static_cast<int>(indent));
    }
    return rc;
}

shared_ptr<ngraph::Function> ngraph::deserialize(istream& in)
{
    std::stringstream ss;
    ss << in.rdbuf();
    return deserialize(ss.str());
}

shared_ptr<ngraph::Function> ngraph::deserialize(const string& s)
{
    json js = json::parse(s);
    shared_ptr<Function> rc;
    unordered_map<string, shared_ptr<Function>> function_map;
    for (json func : js)
    {
        shared_ptr<Function> f = read_function(func, function_map);
        rc = f;
    }

    return rc;
}

static json write(const Function& f)
{
    json function;
    function["name"] = f.get_name();

    for (auto param : f.get_parameters())
    {
        function["parameters"].push_back(param->get_name());
    }
    // TODO Functions can return multiple results
    for (size_t i = 0; i < f.get_output_size(); ++i)
    {
        function["result"].push_back(f.get_output_op(i)->get_name());
    }

    list<shared_ptr<Node>> result_list;
    {
        deque<Node*> independent_nodes;
        unordered_map<const Node*, size_t> node_depencency_count;
        unordered_map<Node*, shared_ptr<Node>> node_map;

        traverse_nodes(const_cast<Function*>(&f), [&](shared_ptr<Node> node) {
            node_map[node.get()] = node;
            node_depencency_count[node.get()] = node->get_input_ops().size();
            if (node->get_input_ops().size() == 0)
            {
                independent_nodes.push_back(node.get());
            }
        });

        while (independent_nodes.size() > 0)
        {
            auto independent_node = independent_nodes.front();
            result_list.push_back(node_map[independent_node]);
            independent_nodes.pop_front();

            for (auto user : independent_node->users())
            {
                node_depencency_count[user] -= 1;
                size_t count = node_depencency_count[user];
                if (count == 0)
                {
                    independent_nodes.push_back(user);
                }
            }
        }
    }

    json nodes;
    for (shared_ptr<Node> node : result_list)
    {
        nodes.push_back(write(*node));
    }
    function["ops"] = nodes;
    return function;
}

static shared_ptr<ngraph::Function>
    read_function(const json& func_js, unordered_map<string, shared_ptr<Function>>& function_map)
{
    shared_ptr<ngraph::Function> rc;

    string func_name = func_js.at("name").get<string>();
    vector<string> func_parameters = func_js.at("parameters").get<vector<string>>();
    vector<string> func_result = func_js.at("result").get<vector<string>>();
    unordered_map<string, shared_ptr<Node>> node_map;
    for (json node_js : func_js.at("ops"))
    {
        string node_name = node_js.at("name").get<string>();
        string node_op = node_js.at("op").get<string>();
        vector<string> node_inputs = node_js.at("inputs").get<vector<string>>();
        vector<string> node_outputs = node_js.at("outputs").get<vector<string>>();
        shared_ptr<Node> node;
        vector<shared_ptr<Node>> args;
        for (const string& name : node_inputs)
        {
            args.push_back(node_map.at(name));
        }

        if (node_op == "Abs")
        {
            node = make_shared<op::Abs>(args[0]);
        }
        else if (node_op == "Acos")
        {
            node = make_shared<op::Acos>(args[0]);
        }
        else if (node_op == "Add")
        {
            node = make_shared<op::Add>(args[0], args[1]);
        }
        else if (node_op == "AllReduce")
        {
            node = make_shared<op::AllReduce>(args[0]);
        }
        else if (node_op == "Asin")
        {
            node = make_shared<op::Asin>(args[0]);
        }
        else if (node_op == "Atan")
        {
            node = make_shared<op::Atan>(args[0]);
        }
        else if (node_op == "AvgPool")
        {
            auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
            auto include_padding_in_avg_computation =
                node_js.at("include_padding_in_avg_computation").get<bool>();
            node = make_shared<op::AvgPool>(args[0],
                                            window_shape,
                                            window_movement_strides,
                                            padding_below,
                                            padding_above,
                                            include_padding_in_avg_computation);
        }
        else if (node_op == "AvgPoolBackprop")
        {
            auto forward_arg_shape = node_js.at("forward_arg_shape").get<vector<size_t>>();
            auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
            auto include_padding_in_avg_computation =
                get_or_default<bool>(node_js, "include_padding_in_avg_computation", false);
            node = make_shared<op::AvgPoolBackprop>(forward_arg_shape,
                                                    args[0],
                                                    window_shape,
                                                    window_movement_strides,
                                                    padding_below,
                                                    padding_above,
                                                    include_padding_in_avg_computation);
        }
        else if (node_op == "BatchNorm")
        {
            auto epsilon = node_js.at("eps").get<double>();
            node = make_shared<op::BatchNorm>(epsilon, args[0], args[1], args[2]);
        }
        else if (node_op == "BatchNormBackprop")
        {
            auto epsilon = node_js.at("eps").get<double>();
            node = make_shared<op::BatchNormBackprop>(
                epsilon, args[0], args[1], args[2], args[3], args[4], args[5]);
        }
        else if (node_op == "Broadcast")
        {
            auto shape = node_js.at("shape").get<vector<size_t>>();
            auto axes = node_js.at("axes").get<set<size_t>>();
            node = make_shared<op::Broadcast>(args[0], shape, axes);
        }
        else if (node_op == "Ceiling")
        {
            node = make_shared<op::Ceiling>(args[0]);
        }
        else if (node_op == "Concat")
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::Concat>(args, axis);
        }
        else if (node_op == "Constant")
        {
            auto type_node_js =
                node_js.count("element_type") == 0 ? node_js.at("value_type") : node_js;
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            auto value = node_js.at("value").get<vector<string>>();
            node = make_shared<op::Constant>(element_type, shape, value);
        }
        else if (node_op == "Convert")
        {
            auto target_type = read_element_type(node_js.at("target_type"));
            node = make_shared<op::Convert>(args[0], target_type);
        }
        else if (node_op == "Convolution")
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();

            // For backwards compatibility, we accept "image_dilation_strides" in place of
            // "data_dilation_strides", and we also allow it to be omitted altogether.
            auto data_dilation_strides_maybe = node_js["data_dilation_strides"];
            if (data_dilation_strides_maybe.empty())
            {
                data_dilation_strides_maybe = node_js["image_dilation_strides"];
            }

            if (data_dilation_strides_maybe.empty())
            {
                node = make_shared<op::Convolution>(args[0],
                                                    args[1],
                                                    window_movement_strides,
                                                    window_dilation_strides,
                                                    padding_below,
                                                    padding_above);
            }
            else
            {
                node = make_shared<op::Convolution>(
                    args[0],
                    args[1],
                    window_movement_strides,
                    window_dilation_strides,
                    padding_below,
                    padding_above,
                    data_dilation_strides_maybe.get<std::vector<size_t>>());
            }
        }
        else if (node_op == "ConvolutionBackpropData")
        {
            auto data_batch_shape = node_js.at("data_batch_shape").get<vector<size_t>>();
            auto window_movement_strides_forward =
                node_js.at("window_movement_strides_forward").get<vector<size_t>>();
            auto window_dilation_strides_forward =
                node_js.at("window_dilation_strides_forward").get<vector<size_t>>();
            auto padding_below_forward =
                node_js.at("padding_below_forward").get<vector<std::ptrdiff_t>>();
            auto padding_above_forward =
                node_js.at("padding_above_forward").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides_forward =
                node_js.at("data_dilation_strides_forward").get<vector<size_t>>();
            node = make_shared<op::ConvolutionBackpropData>(data_batch_shape,
                                                            args[0],
                                                            args[1],
                                                            window_movement_strides_forward,
                                                            window_dilation_strides_forward,
                                                            padding_below_forward,
                                                            padding_above_forward,
                                                            data_dilation_strides_forward);
        }
        else if (node_op == "ConvolutionBackpropFilters")
        {
            auto filters_shape = node_js.at("filters_shape").get<vector<size_t>>();
            auto window_movement_strides_forward =
                node_js.at("window_movement_strides_forward").get<vector<size_t>>();
            auto window_dilation_strides_forward =
                node_js.at("window_dilation_strides_forward").get<vector<size_t>>();
            auto padding_below_forward =
                node_js.at("padding_below_forward").get<vector<std::ptrdiff_t>>();
            auto padding_above_forward =
                node_js.at("padding_above_forward").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides_forward =
                node_js.at("data_dilation_strides_forward").get<vector<size_t>>();
            node = make_shared<op::ConvolutionBackpropFilters>(args[0],
                                                               filters_shape,
                                                               args[1],
                                                               window_movement_strides_forward,
                                                               window_dilation_strides_forward,
                                                               padding_below_forward,
                                                               padding_above_forward,
                                                               data_dilation_strides_forward);
        }
        else if (node_op == "Cos")
        {
            node = make_shared<op::Cos>(args[0]);
        }
        else if (node_op == "Cosh")
        {
            node = make_shared<op::Cosh>(args[0]);
        }
        else if (node_op == "Divide")
        {
            node = make_shared<op::Divide>(args[0], args[1]);
        }
        else if (node_op == "Dot")
        {
            // For backwards compatibility, reduction_axes_count is optional.
            auto obj = node_js["reduction_axes_count"];
            if (obj.empty())
            {
                node = make_shared<op::Dot>(args[0], args[1]);
            }
            else
            {
                size_t reduction_axes_count = obj.get<size_t>();
                node = make_shared<op::Dot>(args[0], args[1], reduction_axes_count);
            }
        }
        else if (node_op == "Equal")
        {
            node = make_shared<op::Equal>(args[0], args[1]);
        }
        else if (node_op == "Exp")
        {
            node = make_shared<op::Exp>(args[0]);
        }
        else if (node_op == "Floor")
        {
            node = make_shared<op::Floor>(args[0]);
        }
        else if (node_op == "FunctionCall")
        {
            string function_name = node_js.at("function").get<string>();
            shared_ptr<Function> f_ptr = function_map.at(function_name);
            node = make_shared<op::FunctionCall>(f_ptr, args);
        }
        else if (node_op == "GetOutputElement")
        {
            node = make_shared<op::GetOutputElement>(args[0], node_js.at("n").get<size_t>());
        }
        else if (node_op == "Greater")
        {
            node = make_shared<op::Greater>(args[0], args[1]);
        }
        else if (node_op == "GreaterEq")
        {
            node = make_shared<op::GreaterEq>(args[0], args[1]);
        }
        else if (node_op == "Less")
        {
            node = make_shared<op::Less>(args[0], args[1]);
        }
        else if (node_op == "LessEq")
        {
            node = make_shared<op::LessEq>(args[0], args[1]);
        }
        else if (node_op == "Log")
        {
            node = make_shared<op::Log>(args[0]);
        }
        else if (node_op == "Max")
        {
            auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
            node = make_shared<op::Max>(args[0], reduction_axes);
        }
        else if (node_op == "MaxPool")
        {
            auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            // For backwards compatibility, both (but not just one) of the padding_ fields may be
            // omitted.
            auto padding_below_maybe = node_js["padding_below"];
            auto padding_above_maybe = node_js["padding_above"];
            if (padding_below_maybe.empty() && !padding_above_maybe.empty())
            {
                throw runtime_error(
                    "MaxPool: padding_below is absent but padding_above is present");
            }
            else if (!padding_below_maybe.empty() && padding_above_maybe.empty())
            {
                throw runtime_error(
                    "MaxPool: padding_below is present but padding_above is absent");
            }
            else if (!padding_below_maybe.empty() && !padding_above_maybe.empty())
            {
                auto padding_below = padding_below_maybe.get<vector<size_t>>();
                auto padding_above = padding_above_maybe.get<vector<size_t>>();
                node = make_shared<op::MaxPool>(
                    args[0], window_shape, window_movement_strides, padding_below, padding_above);
            }
            else
            {
                node = make_shared<op::MaxPool>(args[0], window_shape, window_movement_strides);
            }
        }
        else if (node_op == "MaxPoolBackprop")
        {
            auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
            node = make_shared<op::MaxPoolBackprop>(args[0],
                                                    args[1],
                                                    window_shape,
                                                    window_movement_strides,
                                                    padding_below,
                                                    padding_above);
        }
        else if (node_op == "Maximum")
        {
            node = make_shared<op::Maximum>(args[0], args[1]);
        }
        else if (node_op == "Min")
        {
            auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
            node = make_shared<op::Min>(args[0], reduction_axes);
        }
        else if (node_op == "Minimum")
        {
            node = make_shared<op::Minimum>(args[0], args[1]);
        }
        else if (node_op == "Multiply")
        {
            node = make_shared<op::Multiply>(args[0], args[1]);
        }
        else if (node_op == "Negative")
        {
            node = make_shared<op::Negative>(args[0]);
        }
        else if (node_op == "NotEqual")
        {
            node = make_shared<op::NotEqual>(args[0], args[1]);
        }
        else if (node_op == "Not")
        {
            node = make_shared<op::Not>(args[0]);
        }
        else if (node_op == "OneHot")
        {
            auto shape = node_js.at("shape").get<vector<size_t>>();
            auto one_hot_axis = node_js.at("one_hot_axis").get<size_t>();
            node = make_shared<op::OneHot>(args[0], shape, one_hot_axis);
        }
        else if (node_op == "Pad")
        {
            auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
            auto padding_interior = node_js.at("padding_interior").get<vector<size_t>>();
            node = make_shared<op::Pad>(
                args[0], args[1], padding_below, padding_above, padding_interior);
        }
        else if (node_op == "Parameter")
        {
            auto type_node_js =
                node_js.count("element_type") == 0 ? node_js.at("value_type") : node_js;
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            node = make_shared<op::Parameter>(element_type, shape);
        }
        else if (node_op == "Power")
        {
            node = make_shared<op::Power>(args[0], args[1]);
        }
        else if (node_op == "Product")
        {
            auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
            node = make_shared<op::Product>(args[0], reduction_axes);
        }
        else if (node_op == "Reduce")
        {
            auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
            string function_name = node_js.at("function").get<string>();
            shared_ptr<Function> f_ptr = function_map.at(function_name);
            node = make_shared<op::Reduce>(args[0], args[1], f_ptr, reduction_axes);
        }
        else if (node_op == "ReduceWindow")
        {
            auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            string function_name = node_js.at("function").get<string>();
            shared_ptr<Function> f_ptr = function_map.at(function_name);
            node = make_shared<op::ReduceWindow>(
                args[0], args[1], f_ptr, window_shape, window_movement_strides);
        }
        else if (node_op == "Remainder")
        {
            node = make_shared<op::Remainder>(args[0], args[1]);
        }
        else if (node_op == "Relu")
        {
            node = make_shared<op::Relu>(args[0]);
        }
        else if (node_op == "ReluBackprop")
        {
            node = make_shared<op::ReluBackprop>(args[0], args[1]);
        }
        else if (node_op == "ReplaceSlice")
        {
            auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
            auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::ReplaceSlice>(
                args[0], args[1], lower_bounds, upper_bounds, strides);
        }
        else if (node_op == "Reshape")
        {
            auto input_order = node_js.at("input_order").get<vector<size_t>>();
            auto output_shape = node_js.at("output_shape").get<vector<size_t>>();
            node = make_shared<op::Reshape>(args[0], input_order, output_shape);
        }
        else if (node_op == "Reverse")
        {
            auto reversed_axes = node_js.at("reversed_axes").get<set<size_t>>();
            node = make_shared<op::Reverse>(args[0], reversed_axes);
        }
        else if (node_op == "Select")
        {
            node = make_shared<op::Select>(args[0], args[1], args[2]);
        }
        else if (node_op == "SelectAndScatter")
        {
            string selection_function_name = node_js.at("selection_function").get<string>();
            shared_ptr<Function> selection_f_ptr = function_map.at(selection_function_name);
            string scatter_function_name = node_js.at("scatter_function").get<string>();
            shared_ptr<Function> scatter_f_ptr = function_map.at(scatter_function_name);

            auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();

            node = make_shared<op::SelectAndScatter>(args[0],
                                                     args[1],
                                                     args[2],
                                                     selection_f_ptr,
                                                     scatter_f_ptr,
                                                     window_shape,
                                                     window_movement_strides);
        }
        else if (node_op == "Sign")
        {
            node = make_shared<op::Sign>(args[0]);
        }
        else if (node_op == "Sin")
        {
            node = make_shared<op::Sin>(args[0]);
        }
        else if (node_op == "Sinh")
        {
            node = make_shared<op::Sinh>(args[0]);
        }
        else if (node_op == "Slice")
        {
            auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
            auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::Slice>(args[0], lower_bounds, upper_bounds, strides);
        }
        else if (node_op == "Sqrt")
        {
            node = make_shared<op::Sqrt>(args[0]);
        }
        else if (node_op == "Subtract")
        {
            node = make_shared<op::Subtract>(args[0], args[1]);
        }
        else if (node_op == "Sum")
        {
            auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
            node = make_shared<op::Sum>(args[0], reduction_axes);
        }
        else if (node_op == "Tan")
        {
            node = make_shared<op::Tan>(args[0]);
        }
        else if (node_op == "Tanh")
        {
            node = make_shared<op::Tanh>(args[0]);
        }
        // else if (node_op == "XLAGetTupleElement")
        // {
        // }
        else
        {
            stringstream ss;
            ss << "unsupported op " << node_op;
            throw runtime_error(ss.str());
        }
        node_map[node_name] = node;

        // Typically, it could be unsafe to change the name of a node since it may break nameing
        // uniqueness. However, it could sometimes be helpful to use the original name from
        // the serialization for debugging.
        // node->set_name(node_name);
    }

    std::vector<std::shared_ptr<Node>> result;
    for (auto result_name : func_result)
    {
        result.push_back(node_map.at(result_name));
    }
    std::vector<std::shared_ptr<op::Parameter>> params;
    for (auto param_name : func_parameters)
    {
        params.push_back(dynamic_pointer_cast<op::Parameter>(node_map.at(param_name)));
    }

    rc = make_shared<Function>(result, params, func_name);
    function_map[func_name] = rc;

    return rc;
}

static json write(const Node& n)
{
    json node;
    node["name"] = n.get_name();
    node["op"] = n.description();
    // TODO Multiple outputs
    json inputs = json::array();
    json outputs = json::array();

    for (const descriptor::Input& input : n.get_inputs())
    {
        inputs.push_back(input.get_output().get_node()->get_name());
    }
    for (size_t i = 0; i < n.get_output_size(); ++i)
    {
        outputs.push_back(n.get_output_tensor(i).get_name());
    }

    node["inputs"] = inputs;
    node["outputs"] = outputs;

    if (std::getenv("NGRAPH_SERIALIZER_OUTPUT_SHAPES") != nullptr)
    {
        json output_shapes = json::array();
        for (size_t i = 0; i < n.get_output_size(); ++i)
        {
            output_shapes.push_back(n.get_output_shape(i));
        }
        node["output_shapes"] = output_shapes;
    }

    string node_op = n.description();
    if (node_op == "Abs")
    {
    }
    else if (node_op == "Acos")
    {
    }
    else if (node_op == "Add")
    {
    }
    else if (node_op == "AllReduce")
    {
    }
    else if (node_op == "Asin")
    {
    }
    else if (node_op == "Atan")
    {
    }
    else if (node_op == "AvgPool")
    {
        auto tmp = dynamic_cast<const op::AvgPool*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["include_padding_in_avg_computation"] = tmp->get_include_padding_in_avg_computation();
    }
    else if (node_op == "AvgPoolBackprop")
    {
        auto tmp = dynamic_cast<const op::AvgPoolBackprop*>(&n);
        node["forward_arg_shape"] = tmp->get_forward_arg_shape();
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["include_padding_in_avg_computation"] = tmp->get_include_padding_in_avg_computation();
    }
    else if (node_op == "BatchNorm")
    {
        auto tmp = dynamic_cast<const op::BatchNorm*>(&n);
        node["eps"] = tmp->get_eps_value();
    }
    else if (node_op == "BatchNormBackprop")
    {
        auto tmp = dynamic_cast<const op::BatchNormBackprop*>(&n);
        node["eps"] = tmp->get_eps_value();
    }
    else if (node_op == "Broadcast")
    {
        auto tmp = dynamic_cast<const op::Broadcast*>(&n);
        node["axes"] = tmp->get_broadcast_axes();
        node["shape"] = tmp->get_broadcast_shape();
    }
    else if (node_op == "Ceiling")
    {
    }
    else if (node_op == "Concat")
    {
        auto tmp = dynamic_cast<const op::Concat*>(&n);
        node["axis"] = tmp->get_concatenation_axis();
    }
    else if (node_op == "Constant")
    {
        auto tmp = dynamic_cast<const op::Constant*>(&n);
        node["value"] = tmp->get_value_strings();
        node["shape"] = tmp->get_shape();
        node["element_type"] = write_element_type(tmp->get_element_type());
    }
    else if (node_op == "Convert")
    {
        auto tmp = dynamic_cast<const op::Convert*>(&n);
        node["target_type"] = write_element_type(tmp->get_convert_element_type());
    }
    else if (node_op == "Convolution")
    {
        auto tmp = dynamic_cast<const op::Convolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
    }
    else if (node_op == "ConvolutionBackpropData")
    {
        auto tmp = dynamic_cast<const op::ConvolutionBackpropData*>(&n);
        node["data_batch_shape"] = tmp->get_data_batch_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
    }
    else if (node_op == "ConvolutionBackpropFilters")
    {
        auto tmp = dynamic_cast<const op::ConvolutionBackpropFilters*>(&n);
        node["filters_shape"] = tmp->get_filters_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
    }
    else if (node_op == "Cos")
    {
    }
    else if (node_op == "Cosh")
    {
    }
    else if (node_op == "Divide")
    {
    }
    else if (node_op == "Dot")
    {
        auto tmp = dynamic_cast<const op::Dot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
    }
    else if (node_op == "Equal")
    {
    }
    else if (node_op == "Exp")
    {
    }
    else if (node_op == "Floor")
    {
    }
    else if (node_op == "FunctionCall")
    {
        node["function"] = n.get_functions()[0]->get_name();
    }
    else if (node_op == "GetOutputElement")
    {
        auto tmp = dynamic_cast<const op::GetOutputElement*>(&n);
        node["n"] = tmp->get_n();
    }
    else if (node_op == "Greater")
    {
    }
    else if (node_op == "GreaterEq")
    {
    }
    else if (node_op == "Less")
    {
    }
    else if (node_op == "LessEq")
    {
    }
    else if (node_op == "Log")
    {
    }
    else if (node_op == "Max")
    {
        auto tmp = dynamic_cast<const op::Max*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
    }
    else if (node_op == "MaxPool")
    {
        auto tmp = dynamic_cast<const op::MaxPool*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
    }
    else if (node_op == "MaxPoolBackprop")
    {
        auto tmp = dynamic_cast<const op::MaxPoolBackprop*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
    }
    else if (node_op == "Maximum")
    {
    }
    else if (node_op == "Min")
    {
        auto tmp = dynamic_cast<const op::Min*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
    }
    else if (node_op == "Minimum")
    {
    }
    else if (node_op == "Multiply")
    {
    }
    else if (node_op == "Negative")
    {
    }
    else if (node_op == "NotEqual")
    {
    }
    else if (node_op == "Not")
    {
    }
    else if (node_op == "OneHot")
    {
        auto tmp = dynamic_cast<const op::OneHot*>(&n);
        node["shape"] = tmp->get_shape();
        node["one_hot_axis"] = tmp->get_one_hot_axis();
    }
    else if (node_op == "Pad")
    {
        auto tmp = dynamic_cast<const op::Pad*>(&n);
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["padding_interior"] = tmp->get_padding_interior();
    }
    else if (node_op == "Parameter")
    {
        auto tmp = dynamic_cast<const op::Parameter*>(&n);
        node["shape"] = tmp->get_shape();
        node["element_type"] = write_element_type(tmp->get_element_type());
    }
    else if (node_op == "Product")
    {
        auto tmp = dynamic_cast<const op::Product*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
    }
    else if (node_op == "Power")
    {
    }
    else if (node_op == "Reduce")
    {
        auto tmp = dynamic_cast<const op::Reduce*>(&n);
        node["function"] = tmp->get_functions()[0]->get_name();
        node["reduction_axes"] = tmp->get_reduction_axes();
    }
    else if (node_op == "ReduceWindow")
    {
        auto tmp = dynamic_cast<const op::ReduceWindow*>(&n);
        node["function"] = tmp->get_functions()[0]->get_name();
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
    }
    else if (node_op == "Relu")
    {
    }
    else if (node_op == "ReluBackprop")
    {
    }
    else if (node_op == "Remainder")
    {
    }
    else if (node_op == "ReplaceSlice")
    {
        auto tmp = dynamic_cast<const op::ReplaceSlice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
    }
    else if (node_op == "Reshape")
    {
        auto tmp = dynamic_cast<const op::Reshape*>(&n);
        node["input_order"] = tmp->get_input_order();
        node["output_shape"] = tmp->get_output_shape();
    }
    else if (node_op == "Reverse")
    {
        auto tmp = dynamic_cast<const op::Reverse*>(&n);
        node["reversed_axes"] = tmp->get_reversed_axes();
    }
    else if (node_op == "Select")
    {
    }
    else if (node_op == "SelectAndScatter")
    {
        auto tmp = dynamic_cast<const op::SelectAndScatter*>(&n);
        node["selection_function"] = tmp->get_functions()[0]->get_name();
        node["scatter_function"] = tmp->get_functions()[1]->get_name();
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
    }
    else if (node_op == "Sign")
    {
    }
    else if (node_op == "Sin")
    {
    }
    else if (node_op == "Sinh")
    {
    }
    else if (node_op == "Slice")
    {
        auto tmp = dynamic_cast<const op::Slice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
    }
    else if (node_op == "Sqrt")
    {
    }
    else if (node_op == "Subtract")
    {
    }
    else if (node_op == "Sum")
    {
        auto tmp = dynamic_cast<const op::Sum*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
    }
    else if (node_op == "Tan")
    {
    }
    else if (node_op == "Tanh")
    {
    }
    else if (node_op == "XLAGetTupleElement")
    {
    }

    return node;
}
