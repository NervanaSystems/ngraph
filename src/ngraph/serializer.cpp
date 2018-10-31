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

#include <fstream>
#include <functional>

#include "ngraph/cpio.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/ceiling.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convert.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/cos.hpp"
#include "ngraph/op/cosh.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/function_call.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/log.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/maximum.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/minimum.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/negative.hpp"
#include "ngraph/op/not.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/select_and_scatter.hpp"
#include "ngraph/op/sigmoid.hpp"
#include "ngraph/op/sign.hpp"
#include "ngraph/op/sin.hpp"
#include "ngraph/op/sinh.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sqrt.hpp"
#include "ngraph/op/stop_gradient.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/tan.hpp"
#include "ngraph/op/tanh.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"

using namespace ngraph;
using namespace std;
using json = nlohmann::json;
using const_data_callback_t = shared_ptr<Node>(const string&, const element::Type&, const Shape&);

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/op_tbl.hpp"
    UnknownOp
};
#undef NGRAPH_OP

static OP_TYPEID get_typeid(const string& s)
{
// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// {"Abs", OP_TYPEID::Abs},
// {"Acos", OP_TYPEID::Acos},
// ...
#define NGRAPH_OP(a, b) {#a, OP_TYPEID::a},
    static const unordered_map<string, OP_TYPEID> typeid_map{
#include "ngraph/op/op_tbl.hpp"
    };
#undef NGRAPH_OP
    OP_TYPEID rc = OP_TYPEID::UnknownOp;
    auto it = typeid_map.find(s);
    if (it != typeid_map.end())
    {
        rc = it->second;
    }
    return rc;
}

template <typename T>
T get_or_default(nlohmann::json& j, const std::string& key, const T& default_value)
{
    return j.count(key) != 0 ? j.at(key).get<T>() : default_value;
}

static std::shared_ptr<ngraph::Function>
    read_function(const json&,
                  std::unordered_map<std::string, std::shared_ptr<Function>>&,
                  function<const_data_callback_t>);

static json write(const ngraph::Function&, bool binary_constant_data);
static json write(const ngraph::Node&, bool binary_constant_data);
static string
    serialize(shared_ptr<ngraph::Function> func, size_t indent, bool binary_constant_data);

static json write_dimension(Dimension d)
{
    if (d.is_dynamic())
    {
        return nullptr;
    }
    else
    {
        return static_cast<size_t>(d);
    }
}

static Dimension read_dimension(const json& j)
{
    if (j.is_null())
    {
        return Dimension::dynamic();
    }
    else
    {
        return Dimension(static_cast<size_t>(j));
    }
}

static json write_partial_shape(const PartialShape& s)
{
    if (s.rank().is_dynamic())
    {
        return nullptr;
    }
    else
    {
        std::vector<json> vals(static_cast<size_t>(s.rank()));
        for (size_t i = 0; i < vals.size(); i++)
        {
            vals[i] = write_dimension(s[i]);
        }
        return vals;
    }
}

static PartialShape read_partial_shape(const json& j)
{
    if (j.is_null())
    {
        return PartialShape::dynamic();
    }
    else
    {
        std::vector<Dimension> dims(j.size());
        for (size_t i = 0; i < j.size(); i++)
        {
            dims[i] = read_dimension(j[i]);
        }
        return PartialShape(dims);
    }
}

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
    bool is_quantized = false;
    string c_type_string = "";
    if (j.is_object())
    {
        bitwidth = j.at("bitwidth").get<size_t>();
        is_real = j.at("is_real").get<bool>();
        is_signed = j.at("is_signed").get<bool>();
        is_quantized = j.at("is_quantized").get<bool>();
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
                is_quantized = t->is_quantized();
                c_type_string = t->c_type_string();
                break;
            }
        }
    }
    return element::Type(bitwidth, is_real, is_signed, is_quantized, c_type_string);
}

void ngraph::serialize(const string& path, shared_ptr<ngraph::Function> func, size_t indent)
{
    ofstream out(path);
    serialize(out, func, indent);
}

void ngraph::serialize(ostream& out, shared_ptr<ngraph::Function> func, size_t indent)
{
    string j = ::serialize(func, indent, true);
    cpio::Writer writer(out);
    writer.write(func->get_name(), j.c_str(), static_cast<uint32_t>(j.size()));

    traverse_functions(func, [&](shared_ptr<ngraph::Function> f) {
        traverse_nodes(const_cast<Function*>(f.get()),
                       [&](shared_ptr<Node> node) {
                           if (auto c = dynamic_pointer_cast<op::Constant>(node))
                           {
                               uint32_t size =
                                   static_cast<uint32_t>(shape_size(c->get_output_shape(0)) *
                                                         c->get_output_element_type(0).size());
                               writer.write(c->get_name(), c->get_data_ptr(), size);
                           }
                       },
                       true);
    });
}

static string serialize(shared_ptr<ngraph::Function> func, size_t indent, bool binary_constant_data)
{
    json j;
    vector<json> functions;
    traverse_functions(func, [&](shared_ptr<ngraph::Function> f) {
        functions.push_back(write(*f, binary_constant_data));
    });
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

std::string ngraph::serialize(std::shared_ptr<ngraph::Function> func, size_t indent)
{
    return ::serialize(func, indent, false);
}

shared_ptr<ngraph::Function> ngraph::deserialize(istream& in)
{
    shared_ptr<Function> rc;
    if (cpio::is_cpio(in))
    {
        cpio::Reader reader(in);
        vector<cpio::FileInfo> file_info = reader.get_file_info();
        if (file_info.size() > 0)
        {
            // The first file is the model
            uint32_t size = static_cast<uint32_t>(file_info[0].get_size());
            char* data = new char[size];
            reader.read(file_info[0].get_name(), data, size);
            string jstr(data, size);
            delete[] data;
            json js = json::parse(jstr);
            unordered_map<string, shared_ptr<Function>> function_map;
            for (json func : js)
            {
                shared_ptr<Function> f = read_function(
                    func,
                    function_map,
                    [&](const string& const_name, const element::Type& et, const Shape& shape) {
                        shared_ptr<Node> const_node;
                        for (const cpio::FileInfo& info : file_info)
                        {
                            if (info.get_name() == const_name)
                            {
                                void* const_data = malloc(info.get_size());
                                reader.read(const_name, const_data, info.get_size());
                                const_node = make_shared<op::Constant>(et, shape, const_data);
                                free(const_data);
                                break;
                            }
                        }
                        return const_node;
                    });
                rc = f;
            }
        }
    }
    else
    {
        // json file?
        std::stringstream ss;
        ss << in.rdbuf();
        rc = deserialize(ss.str());
    }
    return rc;
}

shared_ptr<ngraph::Function> ngraph::deserialize(const string& s)
{
    shared_ptr<Function> rc;
    if (file_util::exists(s))
    {
        // s is a file and not a json string
        ifstream in(s, ios_base::binary | ios_base::in);
        rc = deserialize(in);
    }
    else
    {
        json js = json::parse(s);
        unordered_map<string, shared_ptr<Function>> function_map;
        for (json func : js)
        {
            shared_ptr<Function> f = read_function(func, function_map, nullptr);
            rc = f;
        }
    }

    return rc;
}

static json write(const Function& f, bool binary_constant_data)
{
    json function;
    function["name"] = f.get_name();

    vector<string> parameter_list;
    for (auto param : f.get_parameters())
    {
        parameter_list.push_back(param->get_name());
    }
    function["parameters"] = parameter_list;

    // TODO Functions can return multiple results
    for (size_t i = 0; i < f.get_output_size(); ++i)
    {
        function["result"].push_back(f.get_output_op(i)->get_name());
    }

    Function* pf = const_cast<Function*>(&f);
    json nodes;
    for (shared_ptr<Node> node : pf->get_ordered_ops(true))
    {
        nodes.push_back(write(*node, binary_constant_data));
    }

    function["ops"] = nodes;
    return function;
}

static shared_ptr<ngraph::Function>
    read_function(const json& func_js,
                  unordered_map<string, shared_ptr<Function>>& function_map,
                  function<const_data_callback_t> const_data_callback)
{
    shared_ptr<ngraph::Function> rc;

    string func_name = func_js.at("name").get<string>();
    vector<string> func_parameters = func_js.at("parameters").get<vector<string>>();
    vector<string> func_result = func_js.at("result").get<vector<string>>();
    unordered_map<string, shared_ptr<Node>> node_map;
    for (json node_js : func_js.at("ops"))
    {
        try
        {
            string node_name = node_js.at("name").get<string>();
            string node_op = node_js.at("op").get<string>();
            vector<string> node_inputs = node_js.at("inputs").get<vector<string>>();
            vector<string> control_deps_inputs =
                get_or_default<vector<string>>(node_js, "control_deps", vector<string>{});
            vector<string> node_outputs = node_js.at("outputs").get<vector<string>>();
            shared_ptr<Node> node;
            vector<shared_ptr<Node>> args;
            vector<shared_ptr<Node>> control_deps;
            for (const string& name : node_inputs)
            {
                args.push_back(node_map.at(name));
            }
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
            // #pragma GCC diagnostic error "-Wimplicit-fallthrough"
            switch (get_typeid(node_op))
            {
            case OP_TYPEID::Abs:
            {
                node = make_shared<op::Abs>(args[0]);
                break;
            }
            case OP_TYPEID::Acos:
            {
                node = make_shared<op::Acos>(args[0]);
                break;
            }
            case OP_TYPEID::Add:
            {
                node = make_shared<op::Add>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::AllReduce:
            {
                node = make_shared<op::AllReduce>(args[0]);
                break;
            }
            case OP_TYPEID::And:
            {
                node = make_shared<op::And>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::ArgMin:
            {
                auto axis = node_js.at("axis").get<size_t>();
                auto target_type = read_element_type(node_js.at("index_element_type"));
                node = make_shared<op::ArgMin>(args[0], axis, target_type);
                break;
            }
            case OP_TYPEID::ArgMax:
            {
                auto axis = node_js.at("axis").get<size_t>();
                auto target_type = read_element_type(node_js.at("index_element_type"));
                node = make_shared<op::ArgMax>(args[0], axis, target_type);
                break;
            }
            case OP_TYPEID::Asin:
            {
                node = make_shared<op::Asin>(args[0]);
                break;
            }
            case OP_TYPEID::Atan:
            {
                node = make_shared<op::Atan>(args[0]);
                break;
            }
            case OP_TYPEID::AvgPool:
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
                break;
            }
            case OP_TYPEID::AvgPoolBackprop:
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
                break;
            }
            case OP_TYPEID::BatchNormTraining:
            {
                auto epsilon = node_js.at("eps").get<double>();
                node = make_shared<op::BatchNormTraining>(epsilon, args[0], args[1], args[2]);
                break;
            }
            case OP_TYPEID::BatchNormInference:
            {
                auto epsilon = node_js.at("eps").get<double>();
                node = make_shared<op::BatchNormInference>(
                    epsilon, args[0], args[1], args[2], args[3], args[4]);
                break;
            }
            case OP_TYPEID::BatchNormTrainingBackprop:
            {
                auto epsilon = node_js.at("eps").get<double>();
                node = make_shared<op::BatchNormTrainingBackprop>(
                    epsilon, args[0], args[1], args[2], args[3], args[4], args[5]);
                break;
            }
            case OP_TYPEID::Broadcast:
            {
                auto shape = node_js.at("shape").get<vector<size_t>>();
                auto axes = node_js.at("axes").get<set<size_t>>();
                node = make_shared<op::Broadcast>(args[0], shape, axes);
                break;
            }
            case OP_TYPEID::Ceiling:
            {
                node = make_shared<op::Ceiling>(args[0]);
                break;
            }
            case OP_TYPEID::Concat:
            {
                auto axis = node_js.at("axis").get<size_t>();
                node = make_shared<op::Concat>(args, axis);
                break;
            }
            case OP_TYPEID::Constant:
            {
                auto type_node_js =
                    node_js.count("element_type") == 0 ? node_js.at("value_type") : node_js;
                auto element_type = read_element_type(type_node_js.at("element_type"));
                auto shape = type_node_js.at("shape");
                try
                {
                    auto value = node_js.at("value").get<vector<string>>();
                    node = make_shared<op::Constant>(element_type, shape, value);
                }
                catch (...)
                {
                    node = const_data_callback(node_name, element_type, shape);
                }
                break;
            }
            case OP_TYPEID::Convert:
            {
                auto target_type = read_element_type(node_js.at("target_type"));
                node = make_shared<op::Convert>(args[0], target_type);
                break;
            }
            case OP_TYPEID::Convolution:
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
                break;
            }
            case OP_TYPEID::ConvolutionBackpropData:
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
                break;
            }
            case OP_TYPEID::ConvolutionBackpropFilters:
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
                break;
            }
            case OP_TYPEID::Cos:
            {
                node = make_shared<op::Cos>(args[0]);
                break;
            }
            case OP_TYPEID::Cosh:
            {
                node = make_shared<op::Cosh>(args[0]);
                break;
            }
            case OP_TYPEID::Dequantize:
            {
                auto type = read_element_type(node_js.at("type"));
                auto axes = node_js.at("axes").get<set<size_t>>();
                node = make_shared<op::Dequantize>(args[0], args[1], args[2], type, axes);
                break;
            }
            case OP_TYPEID::Divide:
            {
                node = make_shared<op::Divide>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Dot:
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
                break;
            }
            case OP_TYPEID::Equal:
            {
                node = make_shared<op::Equal>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Exp:
            {
                node = make_shared<op::Exp>(args[0]);
                break;
            }
            case OP_TYPEID::Floor:
            {
                node = make_shared<op::Floor>(args[0]);
                break;
            }
            case OP_TYPEID::FunctionCall:
            {
                string function_name = node_js.at("function").get<string>();
                shared_ptr<Function> f_ptr = function_map.at(function_name);
                node = make_shared<op::FunctionCall>(f_ptr, args);
                break;
            }
            case OP_TYPEID::GenerateMask:
            {
                auto output_shape = node_js.at("output_shape").get<vector<size_t>>();
                auto type = read_element_type(node_js.at("type"));
                auto seed = node_js.at("seed").get<unsigned int>();
                auto probability = node_js.at("probability").get<double>();

                node =
                    make_shared<op::GenerateMask>(args[0], output_shape, type, seed, probability);
                break;
            }
            case OP_TYPEID::GetOutputElement:
            {
                node = make_shared<op::GetOutputElement>(args[0], node_js.at("n").get<size_t>());
                break;
            }
            case OP_TYPEID::Greater:
            {
                node = make_shared<op::Greater>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::GreaterEq:
            {
                node = make_shared<op::GreaterEq>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Less:
            {
                node = make_shared<op::Less>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::LessEq:
            {
                node = make_shared<op::LessEq>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Log:
            {
                node = make_shared<op::Log>(args[0]);
                break;
            }
            case OP_TYPEID::LRN:
            {
                auto alpha = node_js.at("alpha").get<double>();
                auto beta = node_js.at("beta").get<double>();
                auto bias = node_js.at("bias").get<double>();
                auto nsize = node_js.at("nsize").get<size_t>();
                node = make_shared<op::LRN>(args[0], alpha, beta, bias, nsize);
                break;
            }
            case OP_TYPEID::Max:
            {
                auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
                node = make_shared<op::Max>(args[0], reduction_axes);
                break;
            }
            case OP_TYPEID::MaxPool:
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
                    node = make_shared<op::MaxPool>(args[0],
                                                    window_shape,
                                                    window_movement_strides,
                                                    padding_below,
                                                    padding_above);
                }
                else
                {
                    node = make_shared<op::MaxPool>(args[0], window_shape, window_movement_strides);
                }
                break;
            }
            case OP_TYPEID::MaxPoolBackprop:
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
                break;
            }
            case OP_TYPEID::Maximum:
            {
                node = make_shared<op::Maximum>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Min:
            {
                auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
                node = make_shared<op::Min>(args[0], reduction_axes);
                break;
            }
            case OP_TYPEID::Minimum:
            {
                node = make_shared<op::Minimum>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Multiply:
            {
                node = make_shared<op::Multiply>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Negative:
            {
                node = make_shared<op::Negative>(args[0]);
                break;
            }
            case OP_TYPEID::NotEqual:
            {
                node = make_shared<op::NotEqual>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Not:
            {
                node = make_shared<op::Not>(args[0]);
                break;
            }
            case OP_TYPEID::OneHot:
            {
                auto shape = node_js.at("shape").get<vector<size_t>>();
                auto one_hot_axis = node_js.at("one_hot_axis").get<size_t>();
                node = make_shared<op::OneHot>(args[0], read_partial_shape(shape), one_hot_axis);
                break;
            }
            case OP_TYPEID::Or:
            {
                node = make_shared<op::Or>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Pad:
            {
                auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
                auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
                auto padding_interior = node_js.at("padding_interior").get<vector<size_t>>();
                node = make_shared<op::Pad>(
                    args[0], args[1], padding_below, padding_above, padding_interior);
                break;
            }
            case OP_TYPEID::Parameter:
            {
                auto type_node_js =
                    node_js.count("element_type") == 0 ? node_js.at("value_type") : node_js;
                auto element_type = read_element_type(type_node_js.at("element_type"));
                auto shape = type_node_js.at("shape");
                auto cacheable = get_or_default<bool>(node_js, "cacheable", false);
                node =
                    make_shared<op::Parameter>(element_type, read_partial_shape(shape), cacheable);
                break;
            }
            case OP_TYPEID::Power:
            {
                node = make_shared<op::Power>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Product:
            {
                auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
                node = make_shared<op::Product>(args[0], reduction_axes);
                break;
            }
            case OP_TYPEID::Quantize:
            {
                auto type = read_element_type(node_js.at("type"));
                auto axes = node_js.at("axes").get<set<size_t>>();
                auto round_mode = node_js.at("round_mode").get<op::Quantize::RoundMode>();
                node = make_shared<op::Quantize>(args[0], args[1], args[2], type, axes, round_mode);
                break;
            }
            case OP_TYPEID::Reduce:
            {
                auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
                string function_name = node_js.at("function").get<string>();
                shared_ptr<Function> f_ptr = function_map.at(function_name);
                node = make_shared<op::Reduce>(args[0], args[1], f_ptr, reduction_axes);
                break;
            }
            case OP_TYPEID::ReduceWindow:
            {
                auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
                auto window_movement_strides =
                    node_js.at("window_movement_strides").get<vector<size_t>>();
                string function_name = node_js.at("function").get<string>();
                shared_ptr<Function> f_ptr = function_map.at(function_name);
                node = make_shared<op::ReduceWindow>(
                    args[0], args[1], f_ptr, window_shape, window_movement_strides);
                break;
            }
            case OP_TYPEID::Relu:
            {
                node = make_shared<op::Relu>(args[0]);
                break;
            }
            case OP_TYPEID::ReluBackprop:
            {
                node = make_shared<op::ReluBackprop>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::ReplaceSlice:
            {
                auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
                auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
                auto strides = node_js.at("strides").get<vector<size_t>>();
                node = make_shared<op::ReplaceSlice>(
                    args[0], args[1], lower_bounds, upper_bounds, strides);
                break;
            }
            case OP_TYPEID::Reshape:
            {
                auto input_order = node_js.at("input_order").get<vector<size_t>>();
                auto output_shape = node_js.at("output_shape").get<vector<size_t>>();
                node = make_shared<op::Reshape>(args[0], input_order, output_shape);
                break;
            }
            case OP_TYPEID::Result:
            {
                node = make_shared<op::Result>(args[0]);
                break;
            }
            case OP_TYPEID::Reverse:
            {
                auto reversed_axes = node_js.at("reversed_axes").get<set<size_t>>();
                node = make_shared<op::Reverse>(args[0], reversed_axes);
                break;
            }
            case OP_TYPEID::ReverseSequence:
            {
                auto batch_axis = node_js.at("batch_axis").get<size_t>();
                auto sequence_axis = node_js.at("sequence_axis").get<size_t>();
                node =
                    make_shared<op::ReverseSequence>(args[0], args[1], batch_axis, sequence_axis);
                break;
            }
            case OP_TYPEID::Select:
            {
                node = make_shared<op::Select>(args[0], args[1], args[2]);
                break;
            }
            case OP_TYPEID::SelectAndScatter:
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
                break;
            }
            case OP_TYPEID::Sigmoid:
            {
                node = make_shared<op::Sigmoid>(args[0]);
                break;
            }
            case OP_TYPEID::SigmoidBackprop:
            {
                node = make_shared<op::SigmoidBackprop>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Sign:
            {
                node = make_shared<op::Sign>(args[0]);
                break;
            }
            case OP_TYPEID::Sin:
            {
                node = make_shared<op::Sin>(args[0]);
                break;
            }
            case OP_TYPEID::Sinh:
            {
                node = make_shared<op::Sinh>(args[0]);
                break;
            }
            case OP_TYPEID::Slice:
            {
                auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
                auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
                auto strides = node_js.at("strides").get<vector<size_t>>();
                node = make_shared<op::Slice>(args[0], lower_bounds, upper_bounds, strides);
                break;
            }
            case OP_TYPEID::Softmax:
            {
                auto softmax_axes = node_js.at("softmax_axes").get<set<size_t>>();
                node = make_shared<op::Softmax>(args[0], softmax_axes);
                break;
            }
            case OP_TYPEID::Sqrt:
            {
                node = make_shared<op::Sqrt>(args[0]);
                break;
            }
            case OP_TYPEID::Subtract:
            {
                node = make_shared<op::Subtract>(args[0], args[1]);
                break;
            }
            case OP_TYPEID::Sum:
            {
                auto reduction_axes = node_js.at("reduction_axes").get<set<size_t>>();
                node = make_shared<op::Sum>(args[0], reduction_axes);
                break;
            }
            case OP_TYPEID::Tan:
            {
                node = make_shared<op::Tan>(args[0]);
                break;
            }
            case OP_TYPEID::Tanh:
            {
                node = make_shared<op::Tanh>(args[0]);
                break;
            }
            case OP_TYPEID::TopK:
            {
                auto top_k_axis = node_js.at("top_k_axis").get<size_t>();
                auto k = node_js.at("k").get<size_t>();
                auto compute_max = node_js.at("compute_max").get<bool>();
                auto target_type = read_element_type(node_js.at("index_element_type"));
                node = make_shared<op::TopK>(args[0], top_k_axis, target_type, k, compute_max);
                break;
            }
            case OP_TYPEID::StopGradient:
            {
                node = make_shared<op::StopGradient>(args[0]);
                break;
            }
            case OP_TYPEID::UnknownOp:
            {
                stringstream ss;
                ss << "unsupported op " << node_op;
                throw runtime_error(ss.str());
            }
            }
#pragma GCC diagnostic pop

            for (const string& name : control_deps_inputs)
            {
                node->add_control_dependency(node_map.at(name));
            }

            node_map[node_name] = node;

            // Typically, it could be unsafe to change the name of a node since it may break nameing
            // uniqueness. However, it could sometimes be helpful to use the original name from
            // the serialization for debugging.
            // node->set_name(node_name);
        }
        catch (...)
        {
            string node_name;
            try
            {
                node_name = node_js.at("name").get<string>();
            }
            catch (...)
            {
                node_name = "UNKNOWN";
            }
            throw runtime_error("Error parsing json at node '" + node_name + "'");
        }
    }

    // This handles both graphs w/ `op::Result` and legacy graphs w/o it
    // If we are dealing w/ a legacy graph, add op::Result for each output node
    ResultVector result;
    size_t results = 0;
    for (auto result_name : func_result)
    {
        auto fr = node_map.at(result_name);
        if (auto res = std::dynamic_pointer_cast<op::Result>(fr))
        {
            result.push_back(res);
            // make sure we have `op::Result` on top of all outputs
            results++;
        }
        else
        {
            result.push_back(std::make_shared<op::Result>(fr));
        }
    }

    if (results != 0 && results != func_result.size())
    {
        throw ngraph_error(
            " Graph serialization is inconsistent. Some op::Results appear to be missing");
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

static json write(const Node& n, bool binary_constant_data)
{
    json node;
    node["name"] = n.get_name();
    node["op"] = n.description();
    // TODO Multiple outputs
    json inputs = json::array();
    json control_deps = json::array();
    json outputs = json::array();

    for (const descriptor::Input& input : n.get_inputs())
    {
        inputs.push_back(input.get_output().get_node()->get_name());
    }
    for (auto cdep : n.get_control_dependencies())
    {
        control_deps.push_back(cdep->get_name());
    }
    for (size_t i = 0; i < n.get_output_size(); ++i)
    {
        outputs.push_back(n.get_output_tensor(i).get_name());
    }

    node["inputs"] = inputs;
    node["control_deps"] = control_deps;
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
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
    // #pragma GCC diagnostic error "-Wimplicit-fallthrough"
    switch (get_typeid(node_op))
    {
    case OP_TYPEID::Abs: { break;
    }
    case OP_TYPEID::Acos: { break;
    }
    case OP_TYPEID::Add: { break;
    }
    case OP_TYPEID::ArgMin:
    {
        auto tmp = dynamic_cast<const op::ArgMin*>(&n);
        node["axis"] = tmp->get_reduction_axis();
        node["index_element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::ArgMax:
    {
        auto tmp = dynamic_cast<const op::ArgMax*>(&n);
        node["axis"] = tmp->get_reduction_axis();
        node["index_element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::AllReduce: { break;
    }
    case OP_TYPEID::And: { break;
    }
    case OP_TYPEID::Asin: { break;
    }
    case OP_TYPEID::Atan: { break;
    }
    case OP_TYPEID::AvgPool:
    {
        auto tmp = dynamic_cast<const op::AvgPool*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["include_padding_in_avg_computation"] = tmp->get_include_padding_in_avg_computation();
        break;
    }
    case OP_TYPEID::AvgPoolBackprop:
    {
        auto tmp = dynamic_cast<const op::AvgPoolBackprop*>(&n);
        node["forward_arg_shape"] = tmp->get_forward_arg_shape();
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["include_padding_in_avg_computation"] = tmp->get_include_padding_in_avg_computation();
        break;
    }
    case OP_TYPEID::BatchNormTraining:
    {
        auto tmp = dynamic_cast<const op::BatchNormTraining*>(&n);
        node["eps"] = tmp->get_eps_value();
        break;
    }
    case OP_TYPEID::BatchNormInference:
    {
        auto tmp = dynamic_cast<const op::BatchNormInference*>(&n);
        node["eps"] = tmp->get_eps_value();
        break;
    }
    case OP_TYPEID::BatchNormTrainingBackprop:
    {
        auto tmp = dynamic_cast<const op::BatchNormTrainingBackprop*>(&n);
        node["eps"] = tmp->get_eps_value();
        break;
    }
    case OP_TYPEID::Broadcast:
    {
        auto tmp = dynamic_cast<const op::Broadcast*>(&n);
        node["axes"] = tmp->get_broadcast_axes();
        node["shape"] = tmp->get_broadcast_shape();
        break;
    }
    case OP_TYPEID::Ceiling: { break;
    }
    case OP_TYPEID::Concat:
    {
        auto tmp = dynamic_cast<const op::Concat*>(&n);
        node["axis"] = tmp->get_concatenation_axis();
        break;
    }
    case OP_TYPEID::Constant:
    {
        auto tmp = dynamic_cast<const op::Constant*>(&n);
        if (!binary_constant_data)
        {
            node["value"] = tmp->get_value_strings();
        }
        node["shape"] = tmp->get_shape();
        node["element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::Convert:
    {
        auto tmp = dynamic_cast<const op::Convert*>(&n);
        node["target_type"] = write_element_type(tmp->get_convert_element_type());
        break;
    }
    case OP_TYPEID::Convolution:
    {
        auto tmp = dynamic_cast<const op::Convolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        break;
    }
    case OP_TYPEID::ConvolutionBackpropData:
    {
        auto tmp = dynamic_cast<const op::ConvolutionBackpropData*>(&n);
        node["data_batch_shape"] = tmp->get_data_batch_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
        break;
    }
    case OP_TYPEID::ConvolutionBackpropFilters:
    {
        auto tmp = dynamic_cast<const op::ConvolutionBackpropFilters*>(&n);
        node["filters_shape"] = tmp->get_filters_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
        break;
    }
    case OP_TYPEID::Cos: { break;
    }
    case OP_TYPEID::Cosh: { break;
    }
    case OP_TYPEID::Dequantize:
    {
        auto tmp = dynamic_cast<const op::Dequantize*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["axes"] = tmp->get_axes();
        break;
    }
    case OP_TYPEID::Divide: { break;
    }
    case OP_TYPEID::Dot:
    {
        auto tmp = dynamic_cast<const op::Dot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
        break;
    }
    case OP_TYPEID::Equal: { break;
    }
    case OP_TYPEID::Exp: { break;
    }
    case OP_TYPEID::Floor: { break;
    }
    case OP_TYPEID::FunctionCall:
    {
        node["function"] = n.get_functions()[0]->get_name();
        break;
    }
    case OP_TYPEID::GetOutputElement:
    {
        auto tmp = dynamic_cast<const op::GetOutputElement*>(&n);
        node["n"] = tmp->get_n();
        break;
    }
    case OP_TYPEID::GenerateMask:
    {
        auto tmp = dynamic_cast<const op::GenerateMask*>(&n);
        node["output_shape"] = tmp->get_shape();
        node["type"] = write_element_type(tmp->get_element_type());
        node["seed"] = tmp->get_seed();
        node["probability"] = tmp->get_probability();
        break;
    }
    case OP_TYPEID::Greater: { break;
    }
    case OP_TYPEID::GreaterEq: { break;
    }
    case OP_TYPEID::Less: { break;
    }
    case OP_TYPEID::LessEq: { break;
    }
    case OP_TYPEID::Log: { break;
    }
    case OP_TYPEID::LRN:
    {
        auto tmp = dynamic_cast<const op::LRN*>(&n);
        node["alpha"] = tmp->get_alpha();
        node["beta"] = tmp->get_beta();
        node["bias"] = tmp->get_bias();
        node["nsize"] = tmp->get_nsize();
        break;
    }
    case OP_TYPEID::Max:
    {
        auto tmp = dynamic_cast<const op::Max*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::MaxPool:
    {
        auto tmp = dynamic_cast<const op::MaxPool*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        break;
    }
    case OP_TYPEID::MaxPoolBackprop:
    {
        auto tmp = dynamic_cast<const op::MaxPoolBackprop*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        break;
    }
    case OP_TYPEID::Maximum: { break;
    }
    case OP_TYPEID::Min:
    {
        auto tmp = dynamic_cast<const op::Min*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::Minimum: { break;
    }
    case OP_TYPEID::Multiply: { break;
    }
    case OP_TYPEID::Negative: { break;
    }
    case OP_TYPEID::NotEqual: { break;
    }
    case OP_TYPEID::Not: { break;
    }
    case OP_TYPEID::OneHot:
    {
        auto tmp = dynamic_cast<const op::OneHot*>(&n);
        node["shape"] = write_partial_shape(tmp->get_output_partial_shape(0));
        node["one_hot_axis"] = tmp->get_one_hot_axis();
        break;
    }
    case OP_TYPEID::Or: { break;
    }
    case OP_TYPEID::Pad:
    {
        auto tmp = dynamic_cast<const op::Pad*>(&n);
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["padding_interior"] = tmp->get_padding_interior();
        break;
    }
    case OP_TYPEID::Parameter:
    {
        auto tmp = dynamic_cast<const op::Parameter*>(&n);
        node["shape"] = write_partial_shape(tmp->get_output_partial_shape(0));
        node["cacheable"] = tmp->get_cacheable();
        node["element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::Product:
    {
        auto tmp = dynamic_cast<const op::Product*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::Power: { break;
    }
    case OP_TYPEID::Quantize:
    {
        auto tmp = dynamic_cast<const op::Quantize*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["axes"] = tmp->get_axes();
        node["round_mode"] = tmp->get_round_mode();
        break;
    }
    case OP_TYPEID::Reduce:
    {
        auto tmp = dynamic_cast<const op::Reduce*>(&n);
        node["function"] = tmp->get_functions()[0]->get_name();
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::ReduceWindow:
    {
        auto tmp = dynamic_cast<const op::ReduceWindow*>(&n);
        node["function"] = tmp->get_functions()[0]->get_name();
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        break;
    }
    case OP_TYPEID::Relu: { break;
    }
    case OP_TYPEID::ReluBackprop: { break;
    }
    case OP_TYPEID::ReplaceSlice:
    {
        auto tmp = dynamic_cast<const op::ReplaceSlice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::Reshape:
    {
        auto tmp = dynamic_cast<const op::Reshape*>(&n);
        node["input_order"] = tmp->get_input_order();
        node["output_shape"] = tmp->get_output_shape();
        break;
    }
    case OP_TYPEID::Result: { break;
    }
    case OP_TYPEID::Reverse:
    {
        auto tmp = dynamic_cast<const op::Reverse*>(&n);
        node["reversed_axes"] = tmp->get_reversed_axes();
        break;
    }
    case OP_TYPEID::ReverseSequence:
    {
        auto tmp = dynamic_cast<const op::ReverseSequence*>(&n);
        node["batch_axis"] = tmp->get_batch_axis();
        node["sequence_axis"] = tmp->get_sequence_axis();
        break;
    }
    case OP_TYPEID::Select: { break;
    }
    case OP_TYPEID::SelectAndScatter:
    {
        auto tmp = dynamic_cast<const op::SelectAndScatter*>(&n);
        node["selection_function"] = tmp->get_functions()[0]->get_name();
        node["scatter_function"] = tmp->get_functions()[1]->get_name();
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        break;
    }
    case OP_TYPEID::Sigmoid: { break;
    }
    case OP_TYPEID::SigmoidBackprop: { break;
    }
    case OP_TYPEID::Sign: { break;
    }
    case OP_TYPEID::Sin: { break;
    }
    case OP_TYPEID::Sinh: { break;
    }
    case OP_TYPEID::Slice:
    {
        auto tmp = dynamic_cast<const op::Slice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::Sqrt: { break;
    }
    case OP_TYPEID::StopGradient: { break;
    }
    case OP_TYPEID::Subtract: { break;
    }
    case OP_TYPEID::Sum:
    {
        auto tmp = dynamic_cast<const op::Sum*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::Softmax:
    {
        auto tmp = dynamic_cast<const op::Softmax*>(&n);
        node["softmax_axes"] = tmp->get_axes();
        break;
    }
    case OP_TYPEID::Tan: { break;
    }
    case OP_TYPEID::Tanh: { break;
    }
    case OP_TYPEID::TopK:
    {
        auto tmp = dynamic_cast<const op::TopK*>(&n);
        node["top_k_axis"] = tmp->get_top_k_axis();
        node["index_element_type"] = write_element_type(tmp->get_index_element_type());
        node["k"] = tmp->get_k();
        node["compute_max"] = tmp->get_compute_max();
        break;
    }
    case OP_TYPEID::UnknownOp: { break;
    }
    }
#pragma GCC diagnostic pop

    return node;
}
