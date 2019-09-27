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

#include <fstream>
#include <functional>
#include <queue>
#include <stack>

#include "ngraph/cpio.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/abs.hpp"
#include "ngraph/op/acos.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/allreduce.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/asin.hpp"
#include "ngraph/op/atan.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/broadcast_distributed.hpp"
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
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/exp.hpp"
#include "ngraph/op/experimental/batch_mat_mul.hpp"
#include "ngraph/op/experimental/compiled_kernel.hpp"
#include "ngraph/op/experimental/dyn_broadcast.hpp"
#include "ngraph/op/experimental/dyn_pad.hpp"
#include "ngraph/op/experimental/dyn_replace_slice.hpp"
#include "ngraph/op/experimental/dyn_reshape.hpp"
#include "ngraph/op/experimental/dyn_slice.hpp"
#include "ngraph/op/experimental/generate_mask.hpp"
#include "ngraph/op/experimental/quantized_conv_bias.hpp"
#include "ngraph/op/experimental/quantized_conv_relu.hpp"
#include "ngraph/op/experimental/quantized_dot_bias.hpp"
#include "ngraph/op/experimental/random_uniform.hpp"
#include "ngraph/op/experimental/range.hpp"
#include "ngraph/op/experimental/shape_of.hpp"
#include "ngraph/op/experimental/tile.hpp"
#include "ngraph/op/experimental/transpose.hpp"
#include "ngraph/op/floor.hpp"
#include "ngraph/op/fused/clamp.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/depth_to_space.hpp"
#include "ngraph/op/fused/elu.hpp"
#include "ngraph/op/fused/fake_quantize.hpp"
#include "ngraph/op/fused/gelu.hpp"
#include "ngraph/op/fused/gemm.hpp"
#include "ngraph/op/fused/grn.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/fused/group_conv_transpose.hpp"
#include "ngraph/op/fused/gru_cell.hpp"
#include "ngraph/op/fused/hard_sigmoid.hpp"
#include "ngraph/op/fused/lstm_cell.hpp"
#include "ngraph/op/fused/matmul.hpp"
#include "ngraph/op/fused/mvn.hpp"
#include "ngraph/op/fused/normalize_l2.hpp"
#include "ngraph/op/fused/prelu.hpp"
#include "ngraph/op/fused/rnn_cell.hpp"
#include "ngraph/op/fused/scale_shift.hpp"
#include "ngraph/op/fused/shuffle_channels.hpp"
#include "ngraph/op/fused/space_to_depth.hpp"
#include "ngraph/op/fused/split.hpp"
#include "ngraph/op/fused/squared_difference.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/gather.hpp"
#include "ngraph/op/gather_nd.hpp"
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
#include "ngraph/op/passthrough.hpp"
#include "ngraph/op/power.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/quantized_convolution.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/recv.hpp"
#include "ngraph/op/reduce_prod.hpp"
#include "ngraph/op/reduce_sum.hpp"
#include "ngraph/op/relu.hpp"
#include "ngraph/op/replace_slice.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/result.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/scatter_add.hpp"
#include "ngraph/op/scatter_nd_add.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/send.hpp"
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
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/xor.hpp"
#include "ngraph/provenance.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"

using namespace ngraph;
using namespace std;
using json = nlohmann::json;
using const_data_callback_t = shared_ptr<Node>(const string&, const element::Type&, const Shape&);

static bool s_serialize_output_shapes_enabled =
    (std::getenv("NGRAPH_SERIALIZER_OUTPUT_SHAPES") != nullptr);

void ngraph::set_serialize_output_shapes(bool enable)
{
    s_serialize_output_shapes_enabled = enable;
}

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/fused_op_tbl.hpp"
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
#include "ngraph/op/fused_op_tbl.hpp"
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

bool has_key(json j, const std::string& key)
{
    return j.count(key) != 0;
}

template <typename T>
T get_or_default(json j, const std::string& key, const T& default_value)
{
    return has_key(j, key) ? j.at(key).get<T>() : default_value;
}

class JSONSerializer
{
public:
    void set_indent(size_t indent) { m_indent = indent; }
    void set_serialize_output_shapes(bool serialize_output_shapes)
    {
        m_serialize_output_shapes = serialize_output_shapes;
    }

    void set_binary_constant_data(bool binary_constant_data)
    {
        m_binary_constant_data = binary_constant_data;
    }

    json serialize_function(const Function& function);
    json serialize_output(const Output<Node>& output);
    json serialize_parameter_vector(const ParameterVector& parameters);
    json serialize_output_vector(const OutputVector& output_vector);
    json serialize_node_reference(const Node& node);
    json serialize_node(const Node& node);
    json serialize_axis_set(const AxisSet& axis_set);

protected:
    size_t m_indent{0};
    bool m_serialize_output_shapes{false};
    bool m_binary_constant_data{false};
    json m_json_nodes;
    set<const Node*> m_nodes_serialized;
    queue<const Node*> m_nodes_to_serialize;
};

class JSONDeserializer
{
public:
    void set_const_data_callback(function<const_data_callback_t> const_data_callback)
    {
        m_const_data_callback = const_data_callback;
    }

    shared_ptr<Function> deserialize_function(json j);
    Output<Node> deserialize_output(json j);
    OutputVector deserialize_output_vector(json j);
    ParameterVector deserialize_parameter_vector(json j);
    shared_ptr<Node> deserialize_node_reference(json j);
    shared_ptr<Node> deserialize_node(json j);
    AxisSet deserialize_axis_set(json j);

protected:
    unordered_map<string, shared_ptr<Node>> m_node_map;
    unordered_map<string, shared_ptr<Function>> m_function_map;
    function<const_data_callback_t> m_const_data_callback;
};

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

static Dimension read_dimension(json j)
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
        return move(vals);
    }
}

static PartialShape read_partial_shape(json j)
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

static json write_auto_broadcast(const op::AutoBroadcastSpec& autob)
{
    json j;
    j["type"] = autob.m_type;
    j["axis"] = autob.m_axis;
    return j;
}

static op::AutoBroadcastSpec read_auto_broadcast(json js_node, const std::string& attr)
{
    if (has_key(js_node, attr))
    {
        json j = js_node[attr];
        return op::AutoBroadcastSpec(static_cast<op::AutoBroadcastType>(j.at("type")),
                                     j.at("axis").get<int64_t>());
    }
    else
    {
        return op::AutoBroadcastSpec();
    }
}

static op::PadType read_pad_type(json node_js)
{
    return has_key(node_js, "pad_type") ? static_cast<op::PadType>(node_js.at("pad_type"))
                                        : op::PadType::EXPLICIT;
}

static op::PadMode read_pad_mode(json node_js)
{
    return has_key(node_js, "pad_mode") ? static_cast<op::PadMode>(node_js.at("pad_mode"))
                                        : op::PadMode::CONSTANT;
}

static op::RoundingType read_rounding_type(json node_js)
{
    return has_key(node_js, "rounding_type")
               ? static_cast<op::RoundingType>(node_js.at("rounding_type"))
               : op::RoundingType::FLOOR;
}

static json write_element_type(const ngraph::element::Type& n)
{
    json j;
    j = n.c_type_string();
    return j;
}

static element::Type read_element_type(json j)
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
    out << ::serialize(func, indent, false);
}

#if defined ENABLE_CPIO_FILE
static void serialize_to_cpio(ostream& out, shared_ptr<ngraph::Function> func, size_t indent)
{
    string j = ::serialize(func, indent, true);
    cpio::Writer writer(out);
    writer.write(func->get_name(), j.c_str(), static_cast<uint32_t>(j.size()));

    traverse_nodes(const_cast<Function*>(func.get()),
                   [&](shared_ptr<Node> node) {
                       if (auto c = node->as_type<op::Constant>())
                       {
                           uint32_t size =
                               static_cast<uint32_t>(shape_size(c->get_output_shape(0)) *
                                                     c->get_output_element_type(0).size());
                           writer.write(c->get_name(), c->get_data_ptr(), size);
                       }
                   },
                   true);
}
#endif

static string serialize(shared_ptr<Function> func, size_t indent, bool binary_constant_data)
{
    JSONSerializer serializer;
    serializer.set_binary_constant_data(binary_constant_data);
    serializer.set_indent(indent);
    serializer.set_serialize_output_shapes(s_serialize_output_shapes_enabled);

    json j;
    j.push_back(serializer.serialize_function(*func));

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
            JSONDeserializer deserializer;
            deserializer.set_const_data_callback(
                [&](const string& const_name, const element::Type& et, const Shape& shape) {
                    shared_ptr<Node> const_node;
                    for (const cpio::FileInfo& info : file_info)
                    {
                        if (info.get_name() == const_name)
                        {
                            void* const_data = ngraph_malloc(info.get_size());
                            reader.read(const_name, const_data, info.get_size());
                            const_node = make_shared<op::Constant>(et, shape, const_data);
                            ngraph_free(const_data);
                            break;
                        }
                    }
                    return const_node;
                });
            for (json func : js)
            {
                rc = deserializer.deserialize_function(func);
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
        JSONDeserializer deserializer;
        for (json func : js)
        {
            rc = deserializer.deserialize_function(func);
        }
    }
    return rc;
}

json JSONSerializer::serialize_parameter_vector(const ParameterVector& parameters)
{
    json json_parameters = json::array();
    for (auto param : parameters)
    {
        json_parameters.push_back(serialize_node_reference(*param));
    }
    return json_parameters;
}

json JSONSerializer::serialize_function(const Function& f)
{
    json function;
    function["name"] = f.get_name();
    function["parameters"] = serialize_parameter_vector(f.get_parameters());

    // TODO Functions can return multiple results
    for (size_t i = 0; i < f.get_output_size(); ++i)
    {
        function["result"].push_back(serialize_node_reference(*f.get_output_op(i)));
    }
    function["ops"] = m_json_nodes;
    return function;
}

template <typename T>
T get_value(json js, const string& key)
{
    T rc = {};
    auto it = js.find(key);
    if (it != js.end())
    {
        rc = it->get<T>();
    }
    return rc;
}

shared_ptr<Node> JSONDeserializer::deserialize_node_reference(json j)
{
    const string& name = j;
    return m_node_map.at(name);
}

Output<Node> JSONDeserializer::deserialize_output(json j)
{
    size_t index;
    json json_node_reference;
    if (j.is_string())
    {
        json_node_reference = j;
        index = 0;
    }
    else if (j.is_object())
    {
        json_node_reference = j["node"];
        index = j["index"];
    }
    else
    {
        throw ngraph_error("Expected string or object an output while deserializing");
    }
    return Output<Node>(deserialize_node_reference(json_node_reference), index);
}

OutputVector JSONDeserializer::deserialize_output_vector(json j)
{
    OutputVector result;
    if (j.is_array())
    {
        for (json jelt : j)
        {
            result.push_back(deserialize_output(jelt));
        }
    }
    return result;
}

json JSONSerializer::serialize_axis_set(const AxisSet& axis_set)
{
    return static_cast<set<size_t>>(axis_set);
}

AxisSet JSONDeserializer::deserialize_axis_set(json j)
{
    AxisSet result;
    if (j.is_array())
    {
        result = j.get<set<size_t>>();
    }
    return result;
}

ParameterVector JSONDeserializer::deserialize_parameter_vector(json json_parameters)
{
    std::vector<std::shared_ptr<op::Parameter>> params;
    for (auto& param_ref : json_parameters)
    {
        params.push_back(as_type_ptr<op::Parameter>(deserialize_node_reference(param_ref)));
    }
    return params;
}

shared_ptr<Function> JSONDeserializer::deserialize_function(json func_js)
{
    string func_name = func_js.at("name").get<string>();
    vector<json> func_result = func_js.at("result");
    for (json node_js : func_js.at("ops"))
    {
        deserialize_node(node_js);
    }

    // This handles both graphs w/ `op::Result` and legacy graphs w/o it
    // If we are dealing w/ a legacy graph, add op::Result for each output node
    ResultVector result;
    size_t results = 0;
    for (auto& result_ref : func_result)
    {
        auto fr = deserialize_node_reference(result_ref);
        if (auto res = as_type_ptr<op::Result>(fr))
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
            "Graph serialization is inconsistent. Some op::Results appear to be missing");
    }

    ParameterVector params = deserialize_parameter_vector(func_js.at("parameters"));

    shared_ptr<Function> rc{make_shared<Function>(result, params, func_name)};
    m_function_map[func_name] = rc;
    return rc;
}

// This helps with conversions to old-style shared-ptr<Node> and new-style Output&
// arguments to node constructors. Uses of OutputHelper should be replaced with Output
// when all op constructors use the new style arguments.
struct OutputHelper
{
    OutputHelper(const Output<Node>& output)
        : m_output(output)
    {
    }

    operator shared_ptr<Node>() const { return get_output_element(m_output); }
    operator const Output<Node>&() const { return m_output; }
    Output<Node> m_output;
};

// This helps with conversions to old-style shared-ptr<Node> and new-style Output&
// arguments to node constructors. Uses of OutputVectorHelper should be replaced with OutputVector
// when all op constructors use the new style arguments.
struct OutputVectorHelper
{
    OutputVectorHelper(const OutputVector& output_vector)
        : m_vector(output_vector)
    {
    }
    OutputVectorHelper() = default;
    OutputHelper operator[](size_t i) const { return OutputHelper(m_vector[i]); }
    void push_back(const Output<Node>& output) { m_vector.push_back(output); }
    size_t size() const { return m_vector.size(); }
    operator vector<shared_ptr<Node>>() const
    {
        vector<shared_ptr<Node>> result;
        for (auto& o : m_vector)
        {
            result.push_back(OutputHelper(o));
        }
        return result;
    }
    operator const OutputVector&() const { return m_vector; }
    OutputVector m_vector;
};

shared_ptr<Node> JSONDeserializer::deserialize_node(json node_js)
{
    shared_ptr<Node> node;
    try
    {
        string node_name = node_js.at("name").get<string>();
        string node_op = node_js.at("op").get<string>();
        string friendly_name = get_value<string>(node_js, "friendly_name");
        size_t op_version = get_value<size_t>(node_js, "op_version");
        vector<json> control_deps_inputs = get_value<vector<json>>(node_js, "control_deps");
        vector<string> node_outputs = get_value<vector<string>>(node_js, "outputs");
        OutputVectorHelper args(deserialize_output_vector(node_js["inputs"]));

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
// #pragma GCC diagnostic error "-Wimplicit-fallthrough"
#endif

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
            node = make_shared<op::Add>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::All:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::All>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::AllReduce:
        {
            node = make_shared<op::AllReduce>(args[0]);
            break;
        }
        case OP_TYPEID::And:
        {
            node = make_shared<op::And>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Any:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::Any>(args[0], reduction_axes);
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
            if (op_version == 0)
            {
                auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
                auto window_movement_strides =
                    node_js.at("window_movement_strides").get<vector<size_t>>();
                auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
                auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
                auto include_padding_in_avg_computation =
                    node_js.at("include_padding_in_avg_computation").get<bool>();
                op::PadType pad_type = read_pad_type(node_js);
                bool ceil_mode = get_or_default<bool>(node_js, "ceil_mode", false);
                node = make_shared<op::v0::AvgPool>(args[0],
                                                    window_shape,
                                                    window_movement_strides,
                                                    padding_below,
                                                    padding_above,
                                                    include_padding_in_avg_computation,
                                                    pad_type,
                                                    ceil_mode);
            }
            if (op_version == 1)
            {
                auto kernel = node_js.at("kernel").get<vector<size_t>>();
                auto strides = node_js.at("strides").get<vector<size_t>>();
                auto pads_begin = node_js.at("pads_begin").get<vector<size_t>>();
                auto pads_end = node_js.at("pads_end").get<vector<size_t>>();
                auto exclude_pad = node_js.at("exclude_pad").get<bool>();
                op::PadType pad_type = read_pad_type(node_js);
                op::RoundingType rounding_type = read_rounding_type(node_js);
                node = make_shared<op::v1::AvgPool>(args[0],
                                                    strides,
                                                    pads_begin,
                                                    pads_end,
                                                    kernel,
                                                    exclude_pad,
                                                    rounding_type,
                                                    pad_type);
            }
            break;
        }
        case OP_TYPEID::AvgPoolBackprop:
        {
            if (op_version == 0)
            {
                auto forward_arg_shape = node_js.at("forward_arg_shape").get<vector<size_t>>();
                auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
                auto window_movement_strides =
                    node_js.at("window_movement_strides").get<vector<size_t>>();
                auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
                auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
                auto include_padding_in_avg_computation =
                    get_or_default<bool>(node_js, "include_padding_in_avg_computation", false);
                node = make_shared<op::v0::AvgPoolBackprop>(forward_arg_shape,
                                                            args[0],
                                                            window_shape,
                                                            window_movement_strides,
                                                            padding_below,
                                                            padding_above,
                                                            include_padding_in_avg_computation);
            }
            if (op_version == 1)
            {
                auto forward_arg_shape = node_js.at("forward_arg_shape").get<vector<size_t>>();
                auto kernel = node_js.at("kernel").get<vector<size_t>>();
                auto strides = node_js.at("strides").get<vector<size_t>>();
                auto pads_begin = node_js.at("pads_begin").get<vector<size_t>>();
                auto pads_end = node_js.at("pads_end").get<vector<size_t>>();
                auto exclude_pad = get_or_default<bool>(node_js, "exclude_pad", true);
                node = make_shared<op::v1::AvgPoolBackprop>(
                    forward_arg_shape, args[0], strides, pads_begin, pads_end, kernel, exclude_pad);
            }
            break;
        }
        case OP_TYPEID::BatchMatMul:
        {
            node = make_shared<op::BatchMatMul>(args[0], args[1]);
            break;
        }

        case OP_TYPEID::BatchNormTraining:
        {
            auto epsilon = node_js.at("eps").get<double>();
            // Odd order for back-compatibility
            node = make_shared<op::BatchNormTraining>(args[2], args[0], args[1], epsilon);
            break;
        }
        case OP_TYPEID::BatchNormInference:
        {
            auto epsilon = node_js.at("eps").get<double>();
            // Odd order for back-compatibility
            node = make_shared<op::BatchNormInference>(
                args[2], args[0], args[1], args[3], args[4], epsilon);
            break;
        }
        case OP_TYPEID::BatchNormTrainingBackprop:
        {
            auto epsilon = node_js.at("eps").get<double>();
            // Odd order for back-compatibility
            node = make_shared<op::BatchNormTrainingBackprop>(
                args[2], args[0], args[1], args[3], args[4], args[5], epsilon);
            break;
        }
        case OP_TYPEID::Broadcast:
        {
            auto shape = node_js.at("shape").get<vector<size_t>>();
            auto axes = deserialize_axis_set(node_js.at("axes"));
            node = make_shared<op::Broadcast>(args[0], shape, axes);
            break;
        }
        case OP_TYPEID::BroadcastDistributed:
        {
            node = make_shared<op::BroadcastDistributed>(args[0]);
            break;
        }
        case OP_TYPEID::BroadcastLike:
        {
            auto initial_axes = deserialize_axis_set(node_js.at("initial_axes"));
            node = make_shared<op::BroadcastLike>(args[0], args[1], initial_axes);
            break;
        }
        case OP_TYPEID::Ceiling:
        {
            node = make_shared<op::Ceiling>(args[0]);
            break;
        }
        case OP_TYPEID::Clamp:
        {
            const auto clamp_min = node_js.at("min").get<float>();
            const auto clamp_max = node_js.at("max").get<float>();
            node = make_shared<op::Clamp>(args[0], clamp_min, clamp_max);
            break;
        }
        case OP_TYPEID::Concat:
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::Concat>(static_cast<OutputVector>(args), axis);
            break;
        }
        case OP_TYPEID::Constant:
        {
            auto type_node_js =
                has_key(node_js, "element_type") ? node_js : node_js.at("value_type");
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            auto value = node_js.at("value").get<vector<string>>();
            node = make_shared<op::Constant>(element_type, shape, value);
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
            json data_dilation_strides;
            if (has_key(node_js, "data_dilation_strides"))
            {
                data_dilation_strides = node_js["data_dilation_strides"];
            }
            else if (has_key(node_js, "image_dilation_strides"))
            {
                data_dilation_strides = node_js["image_dilation_strides"];
            }

            op::PadType pad_type = read_pad_type(node_js);

            if (data_dilation_strides.empty())
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
                node =
                    make_shared<op::Convolution>(args[0],
                                                 args[1],
                                                 window_movement_strides,
                                                 window_dilation_strides,
                                                 padding_below,
                                                 padding_above,
                                                 data_dilation_strides.get<std::vector<size_t>>(),
                                                 pad_type);
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
        case OP_TYPEID::ConvolutionBias:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();

            node = make_shared<op::ConvolutionBias>(args[0],
                                                    args[1],
                                                    args[2],
                                                    window_movement_strides,
                                                    window_dilation_strides,
                                                    padding_below,
                                                    padding_above,
                                                    data_dilation_strides);
            break;
        }
        case OP_TYPEID::ConvolutionBiasAdd:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();

            node = make_shared<op::ConvolutionBiasAdd>(args[0],
                                                       args[1],
                                                       args[2],
                                                       args[3],
                                                       window_movement_strides,
                                                       window_dilation_strides,
                                                       padding_below,
                                                       padding_above,
                                                       data_dilation_strides);
            break;
        }
        case OP_TYPEID::ConvolutionBiasBackpropFiltersBias:
        {
            auto filters_shape = node_js.at("filters_shape").get<vector<size_t>>();
            auto bias_shape = node_js.at("bias_shape").get<vector<size_t>>();
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
            node =
                make_shared<op::ConvolutionBiasBackpropFiltersBias>(args[0],
                                                                    filters_shape,
                                                                    bias_shape,
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
        case OP_TYPEID::DepthToSpace:
        {
            auto block_size = node_js.at("block_size").get<size_t>();
            node = make_shared<op::DepthToSpace>(args[0], block_size);
            break;
        }
        case OP_TYPEID::Dequantize:
        {
            auto type = read_element_type(node_js.at("type"));
            auto axes = deserialize_axis_set(node_js.at("axes"));
            node = make_shared<op::Dequantize>(args[0], args[1], args[2], type, axes);
            break;
        }
        case OP_TYPEID::Divide:
        {
            bool pythondiv = get_or_default(node_js, "pythondiv", true);
            node = make_shared<op::Divide>(
                args[0], args[1], pythondiv, read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Dot:
        {
            // For backwards compatibility, reduction_axes_count is optional.
            if (has_key(node_js, "reduction_axes_count"))
            {
                size_t reduction_axes_count = node_js["reduction_axes_count"].get<size_t>();
                node = make_shared<op::Dot>(args[0], args[1], reduction_axes_count);
            }
            else
            {
                node = make_shared<op::Dot>(args[0], args[1]);
            }
            break;
        }
        case OP_TYPEID::DynBroadcast:
        {
            node = make_shared<op::DynBroadcast>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::DynPad:
        {
            node = make_shared<op::DynPad>(args[0], args[1], args[2], args[3]);
            break;
        }
        case OP_TYPEID::DynReplaceSlice:
        {
            auto lower_bounds_mask = node_js.at("lower_bounds_mask").get<set<size_t>>();
            auto upper_bounds_mask = node_js.at("upper_bounds_mask").get<set<size_t>>();
            auto new_axis = node_js.at("new_axis").get<set<size_t>>();
            auto shrink_axis = node_js.at("shrink_axis").get<set<size_t>>();
            auto ellipsis_mask = node_js.at("ellipsis_mask").get<set<size_t>>();
            node = make_shared<op::DynReplaceSlice>(args[0],
                                                    args[1],
                                                    args[2],
                                                    args[3],
                                                    args[4],
                                                    lower_bounds_mask,
                                                    upper_bounds_mask,
                                                    new_axis,
                                                    shrink_axis,
                                                    ellipsis_mask);
            break;
        }
        case OP_TYPEID::DynReshape:
        {
            node = make_shared<op::DynReshape>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::DynSlice:
        {
            auto lower_bounds_mask = node_js.at("lower_bounds_mask").get<set<size_t>>();
            auto upper_bounds_mask = node_js.at("upper_bounds_mask").get<set<size_t>>();
            auto new_axis = node_js.at("new_axis").get<set<size_t>>();
            auto shrink_axis = node_js.at("shrink_axis").get<set<size_t>>();
            auto ellipsis_mask = node_js.at("ellipsis_mask").get<set<size_t>>();
            node = make_shared<op::DynSlice>(args[0],
                                             args[1],
                                             args[2],
                                             args[3],
                                             lower_bounds_mask,
                                             upper_bounds_mask,
                                             new_axis,
                                             shrink_axis,
                                             ellipsis_mask);
            break;
        }
        case OP_TYPEID::Elu:
        {
            auto alpha = node_js.at("alpha").get<double>();
            node = make_shared<op::Elu>(args[0], alpha);
            break;
        }
        case OP_TYPEID::EmbeddingLookup:
        {
            node = make_shared<op::EmbeddingLookup>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Equal:
        {
            node = make_shared<op::Equal>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Erf:
        {
            node = make_shared<op::Erf>(args[0]);
            break;
        }
        case OP_TYPEID::Exp:
        {
            node = make_shared<op::Exp>(args[0]);
            break;
        }
        case OP_TYPEID::FakeQuantize:
        {
            size_t levels = node_js.at("levels").get<size_t>();
            node =
                make_shared<op::FakeQuantize>(args[0], args[1], args[2], args[3], args[4], levels);
            break;
        }
        case OP_TYPEID::Floor:
        {
            node = make_shared<op::Floor>(args[0]);
            break;
        }
        case OP_TYPEID::Gather:
        {
            if (op_version == 0)
            {
                auto axis = node_js.at("axis").get<size_t>();
                node = make_shared<op::v0::Gather>(args[0], args[1], axis);
            }
            if (op_version == 1)
            {
                node = make_shared<op::v1::Gather>(args[0], args[1], args[2]);
            }
            break;
        }
        case OP_TYPEID::GatherND:
        {
            node = make_shared<op::GatherND>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Gelu:
        {
            node = make_shared<op::Gelu>(args[0]);
            break;
        }
        case OP_TYPEID::GeluBackpropFactor:
        {
            node = make_shared<op::GeluBackpropFactor>(args[0]);
            break;
        }
        case OP_TYPEID::Gemm:
        {
            auto alpha = node_js.at("alpha").get<double>();
            auto beta = node_js.at("beta").get<double>();
            auto transA = node_js.at("transA").get<bool>();
            auto transB = node_js.at("transB").get<bool>();
            node = make_shared<op::Gemm>(args[0], args[1], args[2], alpha, beta, transA, transB);
            break;
        }
        case OP_TYPEID::GenerateMask:
        {
            auto output_shape = node_js.at("output_shape").get<vector<size_t>>();
            auto type = read_element_type(node_js.at("type"));
            auto seed = node_js.at("seed").get<unsigned int>();
            auto probability = node_js.at("probability").get<double>();
            bool use_seed = get_or_default<bool>(node_js, "use_seed", false);

            node = make_shared<op::GenerateMask>(
                args[0], output_shape, type, seed, probability, use_seed);
            break;
        }
        case OP_TYPEID::GetOutputElement:
        {
            node = make_shared<op::GetOutputElement>(
                static_cast<Output<Node>>(args[0]).get_node_shared_ptr(),
                node_js.at("n").get<size_t>());
            break;
        }
        case OP_TYPEID::Greater:
        {
            node = make_shared<op::Greater>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            node = make_shared<op::GreaterEq>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::GRN:
        {
            auto bias = node_js.at("bias").get<float>();
            node = make_shared<op::GRN>(args[0], bias);
            break;
        }
        case OP_TYPEID::GroupConvolution:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();
            auto groups = node_js.at("groups").get<size_t>();

            op::PadType pad_type = read_pad_type(node_js);
            node = make_shared<op::GroupConvolution>(args[0],
                                                     args[1],
                                                     window_movement_strides,
                                                     window_dilation_strides,
                                                     padding_below,
                                                     padding_above,
                                                     data_dilation_strides,
                                                     groups,
                                                     pad_type);
            break;
        }
        case OP_TYPEID::GroupConvolutionTranspose:
        {
            auto strides = node_js.at("strides").get<vector<size_t>>();
            auto dilations = node_js.at("dilations").get<vector<size_t>>();
            auto padding_begin = node_js.at("padding_begin").get<vector<ptrdiff_t>>();
            auto padding_end = node_js.at("padding_end").get<vector<ptrdiff_t>>();
            auto output_padding = node_js.at("output_padding").get<vector<ptrdiff_t>>();
            auto groups = node_js.at("groups").get<size_t>();
            op::PadType pad_type = read_pad_type(node_js);
            auto output_shape = node_js.at("output_shape").get<vector<size_t>>();

            node = make_shared<op::GroupConvolutionTranspose>(args[0],
                                                              args[1],
                                                              strides,
                                                              dilations,
                                                              padding_begin,
                                                              padding_end,
                                                              output_padding,
                                                              groups,
                                                              pad_type,
                                                              output_shape);
            break;
        }
        case OP_TYPEID::GRUCell:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto clip = node_js.at("clip").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activation_alpha = node_js.at("activation_alpha").get<vector<float>>();
            auto activation_beta = node_js.at("activation_beta").get<vector<float>>();
            auto linear_before_reset = node_js.at("linear_before_reset").get<bool>();
            node = make_shared<op::GRUCell>(args[0],
                                            args[1],
                                            args[2],
                                            args[3],
                                            hidden_size,
                                            args[4],
                                            activations,
                                            activation_alpha,
                                            activation_beta,
                                            clip,
                                            linear_before_reset);
            break;
        }
        case OP_TYPEID::HardSigmoid:
        {
            auto alpha = node_js.at("alpha").get<float>();
            auto beta = node_js.at("beta").get<float>();
            node = make_shared<op::HardSigmoid>(args[0], alpha, beta);
            break;
        }

        case OP_TYPEID::Less:
        {
            node = make_shared<op::Less>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::LessEq:
        {
            node = make_shared<op::LessEq>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
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
            node = make_shared<op::LRN>(args[0], args[1], alpha, beta, bias, nsize);
            break;
        }
        case OP_TYPEID::LSTMCell:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto clip = node_js.at("clip").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activation_alpha = node_js.at("activation_alpha").get<vector<float>>();
            auto activation_beta = node_js.at("activation_beta").get<vector<float>>();
            auto input_forget = node_js.at("input_forget").get<bool>();
            node = make_shared<op::LSTMCell>(args[0],
                                             args[1],
                                             args[2],
                                             args[3],
                                             args[4],
                                             hidden_size,
                                             args[5],
                                             args[6],
                                             activations,
                                             activation_alpha,
                                             activation_beta,
                                             clip,
                                             input_forget);
            break;
        }
        case OP_TYPEID::MatMul:
        {
            bool transpose_a = node_js.at("transpose_a").get<bool>();
            bool transpose_b = node_js.at("transpose_b").get<bool>();
            node = make_shared<op::MatMul>(args[0], args[1], transpose_a, transpose_b);
            break;
        }
        case OP_TYPEID::Max:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::Max>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::MaxPool:
        {
            if (op_version == 0)
            {
                auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
                auto window_movement_strides =
                    node_js.at("window_movement_strides").get<vector<size_t>>();
                // For backwards compatibility, both (but not just one) of the padding_ fields may
                // be omitted.
                auto padding_below_maybe = get_or_default(node_js, "padding_below", json{});
                auto padding_above_maybe = get_or_default(node_js, "padding_above", json{});
                op::PadType pad_type = read_pad_type(node_js);
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
                    node = make_shared<op::v0::MaxPool>(args[0],
                                                        window_shape,
                                                        window_movement_strides,
                                                        padding_below,
                                                        padding_above,
                                                        pad_type);
                }
                else
                {
                    node = make_shared<op::v0::MaxPool>(
                        args[0], window_shape, window_movement_strides);
                }
            }
            if (op_version == 1)
            {
                auto kernel = node_js.at("kernel").get<vector<size_t>>();
                auto strides = node_js.at("strides").get<vector<size_t>>();
                auto pads_begin = node_js.at("pads_begin").get<vector<size_t>>();
                auto pads_end = node_js.at("pads_end").get<vector<size_t>>();
                auto rounding_type = read_rounding_type(node_js);
                op::PadType pad_type = read_pad_type(node_js);
                node = make_shared<op::v1::MaxPool>(
                    args[0], strides, pads_begin, pads_end, kernel, rounding_type, pad_type);
            }
            break;
        }
        case OP_TYPEID::MaxPoolBackprop:
        {
            if (op_version == 0)
            {
                auto window_shape = node_js.at("window_shape").get<vector<size_t>>();
                auto window_movement_strides =
                    node_js.at("window_movement_strides").get<vector<size_t>>();
                auto padding_below = node_js.at("padding_below").get<vector<size_t>>();
                auto padding_above = node_js.at("padding_above").get<vector<size_t>>();
                if (args.size() == 3)
                {
                    node = make_shared<op::v0::MaxPoolBackprop>(args[0],
                                                                args[1],
                                                                args[2],
                                                                window_shape,
                                                                window_movement_strides,
                                                                padding_below,
                                                                padding_above);
                }
                else
                {
                    node = make_shared<op::v0::MaxPoolBackprop>(args[0],
                                                                args[1],
                                                                window_shape,
                                                                window_movement_strides,
                                                                padding_below,
                                                                padding_above);
                }
            }
            if (op_version == 1)
            {
                auto kernel = node_js.at("kernel").get<vector<size_t>>();
                auto strides = node_js.at("strides").get<vector<size_t>>();
                auto pads_begin = node_js.at("pads_begin").get<vector<size_t>>();
                auto pads_end = node_js.at("pads_end").get<vector<size_t>>();
                if (args.size() == 3)
                {
                    node = make_shared<op::v1::MaxPoolBackprop>(
                        args[0], args[1], args[2], kernel, strides, pads_begin, pads_end);
                }
                else
                {
                    node = make_shared<op::v1::MaxPoolBackprop>(
                        args[0], args[1], kernel, strides, pads_begin, pads_end);
                }
            }
            break;
        }
        case OP_TYPEID::Maximum:
        {
            node = make_shared<op::Maximum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Min:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::Min>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::Minimum:
        {
            node = make_shared<op::Minimum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Multiply:
        {
            node = make_shared<op::Multiply>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::MVN:
        {
            auto normalize_variance = node_js.at("normalize_variance").get<bool>();
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            auto eps = node_js.at("eps").get<double>();
            node = make_shared<op::MVN>(args[0], normalize_variance, normalize_variance, eps);
            break;
        }
        case OP_TYPEID::Negative:
        {
            node = make_shared<op::Negative>(args[0]);
            break;
        }
        case OP_TYPEID::NormalizeL2:
        {
            float eps = node_js.at("eps").get<float>();
            auto eps_mode = node_js.at("eps_mode").get<op::EpsMode>();
            node = make_shared<op::NormalizeL2>(args[0], args[1], eps, eps_mode);
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            node = make_shared<op::NotEqual>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
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
            node = make_shared<op::Or>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Pad:
        {
            if (op_version == 0)
            {
                auto padding_below = node_js.at("padding_below").get<vector<ptrdiff_t>>();
                auto padding_above = node_js.at("padding_above").get<vector<ptrdiff_t>>();

                // This is a legacy field whose functionality is no longer supported. The new
                // behavior is equivalent to interior padding of 0, so we will accept it under
                // those conditions.
                auto padding_interior = get_value<vector<size_t>>(node_js, "padding_interior");
                NGRAPH_CHECK(std::all_of(padding_interior.begin(),
                                         padding_interior.end(),
                                         [](size_t s) { return s == 0; }),
                             "Legacy padding_interior field must be zero everywhere.");

                auto pad_mode = read_pad_mode(node_js);

                node = make_shared<op::v0::Pad>(
                    args[0], args[1], padding_below, padding_above, pad_mode);
            }
            if (op_version == 1)
            {
                auto pad_mode = read_pad_mode(node_js);
                if (args.size() == 4)
                {
                    node = make_shared<op::v1::Pad>(args[0], args[1], args[2], args[3], pad_mode);
                }
                else
                {
                    node = make_shared<op::v1::Pad>(args[0], args[1], args[2], pad_mode);
                }
            }
            break;
        }
        case OP_TYPEID::Parameter:
        {
            auto type_node_js =
                has_key(node_js, "element_type") ? node_js : node_js.at("value_type");
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            auto cacheable = get_or_default<bool>(node_js, "cacheable", false);
            node = make_shared<op::Parameter>(element_type, read_partial_shape(shape), cacheable);
            break;
        }
        case OP_TYPEID::Passthrough:
        {
            std::vector<json> outputs_js = node_js.at("output_shapes");
            std::vector<std::tuple<element::Type, PartialShape>> outputs;
            for (auto output_js : outputs_js)
            {
                outputs.emplace_back(read_element_type(output_js.at("element_type")),
                                     read_partial_shape(output_js.at("shape")));
            }
            node = make_shared<op::Passthrough>(node_js.at("logical_type"),
                                                node_js.at("language"),
                                                node_js.at("function"),
                                                static_cast<OutputVector>(args),
                                                std::move(outputs));
            break;
        }
        case OP_TYPEID::Power:
        {
            node = make_shared<op::Power>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::PRelu:
        {
            node = make_shared<op::PRelu>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Product:
        {
            if (op_version == 0)
            {
                auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
                if (reduction_axes.empty())
                    node = make_shared<op::v0::Product>(args[0], args[1]);
                else
                    node = make_shared<op::v0::Product>(args[0], reduction_axes);
            }
            if (op_version == 1)
            {
                auto keep_dims = node_js.at("keep_dims").get<bool>();
                node = make_shared<op::v1::ReduceProd>(args[0], args[1], keep_dims);
            }
            break;
        }
        case OP_TYPEID::Quantize:
        {
            auto type = read_element_type(node_js.at("type"));
            auto axes = deserialize_axis_set(node_js.at("axes"));
            auto round_mode = node_js.at("round_mode").get<op::Quantize::RoundMode>();
            node = make_shared<op::Quantize>(args[0], args[1], args[2], type, axes, round_mode);
            break;
        }
        case OP_TYPEID::QuantizedConvolutionBias: { break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasAdd: { break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd: { break;
        }
        case OP_TYPEID::QuantizedConvolutionRelu: { break;
        }
        case OP_TYPEID::QuantizedConvolution:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js["data_dilation_strides"];
            auto output_type = read_element_type(node_js.at("output_type"));
            auto input_axes = node_js.at("input_axes").get<set<size_t>>();
            auto filter_axes = node_js.at("filter_axes").get<set<size_t>>();
            auto output_axes = node_js.at("output_axes").get<set<size_t>>();
            node = make_shared<op::QuantizedConvolution>(
                args[0],
                args[1],
                window_movement_strides,
                window_dilation_strides,
                padding_below,
                padding_above,
                data_dilation_strides.get<std::vector<size_t>>(),
                args[2],
                args[3],
                args[4],
                args[5],
                args[6],
                args[7],
                output_type,
                input_axes,
                filter_axes,
                output_axes);

            break;
        }
        case OP_TYPEID::QuantizedDotBias: { break;
        }
        case OP_TYPEID::QuantizedDot:
        {
            size_t reduction_axes_count = node_js["reduction_axes_count"].get<size_t>();
            auto output_type = read_element_type(node_js.at("output_type"));
            auto input0_axes = node_js.at("input0_axes").get<set<size_t>>();
            auto input1_axes = node_js.at("input1_axes").get<set<size_t>>();
            auto output_axes = node_js.at("output_axes").get<set<size_t>>();

            node = make_shared<op::QuantizedDot>(args[0],
                                                 args[1],
                                                 reduction_axes_count,
                                                 args[2],
                                                 args[3],
                                                 args[4],
                                                 args[5],
                                                 args[6],
                                                 args[7],
                                                 output_type,
                                                 input0_axes,
                                                 input1_axes,
                                                 output_axes);

            break;
        }
        case OP_TYPEID::Recv:
        {
            auto src_id = node_js.at("source_id").get<size_t>();
            node = make_shared<op::Recv>(args[0], src_id);
            break;
        }
        case OP_TYPEID::RandomUniform:
        {
            auto fixed_seed = node_js.at("fixed_seed").get<uint64_t>();
            node = make_shared<op::RandomUniform>(args[0], args[1], args[2], args[3], fixed_seed);
            break;
        }
        case OP_TYPEID::Range:
        {
            node = make_shared<op::Range>(args[0], args[1], args[2]);
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
            auto needs_default_layout =
                get_or_default<bool>(node_js, "needs_default_layout", false);
            node = make_shared<op::Result>(args[0], needs_default_layout);
            break;
        }
        case OP_TYPEID::Reverse:
        {
            auto reversed_axes = deserialize_axis_set(node_js.at("reversed_axes"));
            node = make_shared<op::Reverse>(args[0], reversed_axes);
            break;
        }
        case OP_TYPEID::ReverseSequence:
        {
            auto batch_axis = node_js.at("batch_axis").get<size_t>();
            auto sequence_axis = node_js.at("sequence_axis").get<size_t>();
            node = make_shared<op::ReverseSequence>(args[0], args[1], batch_axis, sequence_axis);
            break;
        }
        case OP_TYPEID::RNNCell:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto clip = node_js.at("clip").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activation_alpha = node_js.at("activation_alpha").get<vector<float>>();
            auto activation_beta = node_js.at("activation_beta").get<vector<float>>();
            node = make_shared<op::RNNCell>(args[0],
                                            args[1],
                                            args[2],
                                            args[3],
                                            hidden_size,
                                            args[4],
                                            activations,
                                            activation_alpha,
                                            activation_beta,
                                            clip);
            break;
        }
        case OP_TYPEID::ScalarConstantLike:
        {
            double value = node_js.at("value").get<double>();
            node = make_shared<op::ScalarConstantLike>(args[0], value);
            break;
        }
        case OP_TYPEID::ScaleShift:
        {
            node = make_shared<op::ScaleShift>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterAdd:
        {
            node = make_shared<op::ScatterAdd>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterNDAdd:
        {
            node = make_shared<op::ScatterNDAdd>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Select:
        {
            node = make_shared<op::Select>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Send:
        {
            auto dest_id = node_js.at("dest_id").get<size_t>();
            node = make_shared<op::Send>(args[0], dest_id);
            break;
        }
        case OP_TYPEID::ShapeOf:
        {
            node = make_shared<op::ShapeOf>(args[0]);
            break;
        }
        case OP_TYPEID::ShuffleChannels:
        {
            const auto axis = node_js.at("axis").get<size_t>();
            const auto groups = node_js.at("groups").get<size_t>();
            node = make_shared<op::ShuffleChannels>(args[0], axis, groups);
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
            if (op_version == 0)
            {
                auto softmax_axes = deserialize_axis_set(node_js.at("softmax_axes"));
                node = make_shared<op::Softmax>(args[0], softmax_axes);
            }
            if (op_version == 1)
            {
                size_t softmax_axis = node_js.at("softmax_axis");
                node = make_shared<op::v1::Softmax>(args[0], softmax_axis);
            }
            break;
        }
        case OP_TYPEID::SpaceToDepth:
        {
            auto block_size = node_js.at("block_size").get<size_t>();
            node = make_shared<op::SpaceToDepth>(args[0], block_size);
            break;
        }
        case OP_TYPEID::Split:
        {
            const auto axis = node_js.at("axis").get<size_t>();
            const auto splits = node_js.at("splits").get<vector<size_t>>();
            node = make_shared<op::Split>(args[0], axis, splits);
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            node = make_shared<op::Sqrt>(args[0]);
            break;
        }
        case OP_TYPEID::SquaredDifference:
        {
            node = make_shared<op::SquaredDifference>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Squeeze:
        {
            node = make_shared<op::Squeeze>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Subtract:
        {
            node = make_shared<op::Subtract>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Sum:
        {
            if (op_version == 0)
            {
                auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
                if (reduction_axes.empty())
                    node = make_shared<op::v0::Sum>(args[0], args[1]);
                else
                    node = make_shared<op::v0::Sum>(args[0], reduction_axes);
            }
            if (op_version == 1)
            {
                auto keep_dims = node_js.at("keep_dims").get<bool>();
                node = make_shared<op::v1::ReduceSum>(args[0], args[1], keep_dims);
            }
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
        case OP_TYPEID::Tile:
        {
            node = make_shared<op::Tile>(args[0], args[1]);
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
        case OP_TYPEID::Transpose:
        {
            node = make_shared<op::Transpose>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::StopGradient:
        {
            node = make_shared<op::StopGradient>(args[0]);
            break;
        }
        case OP_TYPEID::Unsqueeze:
        {
            node = make_shared<op::Unsqueeze>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Xor:
        {
            node = make_shared<op::Xor>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::UnknownOp:
        {
            stringstream ss;
            ss << "unsupported op " << node_op;
            throw runtime_error(ss.str());
        }
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

        for (auto& control_dep : control_deps_inputs)
        {
            node->add_control_dependency(deserialize_node_reference(control_dep));
        }

        if (!friendly_name.empty())
        {
            node->set_friendly_name(friendly_name);
        }
        else
        {
            node->set_friendly_name(node_name);
        }
        if (ngraph::get_provenance_enabled())
        {
            std::vector<json> prov_js = node_js.at("provenance_tags");
            for (auto prov_tag : prov_js)
            {
                node->add_provenance_tag(prov_tag);
            }
        }
        m_node_map[node_name] = node;
    }
    catch (...)
    {
        string node_name;
        auto it = node_js.find("name");
        if (it != node_js.end())
        {
            node_name = it->get<string>();
        }
        else
        {
            node_name = "UNKNOWN";
        }
        throw runtime_error("Error parsing json at node '" + node_name + "'");
    }
    return node;
}

json JSONSerializer::serialize_node_reference(const Node& n)
{
    if (m_nodes_serialized.count(&n) != 1)
    {
        m_nodes_to_serialize.push(&n);
        if (m_nodes_to_serialize.size() == 1)
        {
            // Nothing in the queue
            stack<json> serialized_nodes;
            while (!m_nodes_to_serialize.empty())
            {
                const Node* next_node = m_nodes_to_serialize.front();
                m_nodes_to_serialize.pop();
                serialized_nodes.push(serialize_node(*next_node));
            }
            while (serialized_nodes.size() > 0)
            {
                m_json_nodes.push_back(serialized_nodes.top());
                serialized_nodes.pop();
            }
        }
    }
    return n.get_name();
}

json JSONSerializer::serialize_output(const Output<Node>& output)
{
    json result;
    auto index = output.get_index();
    json json_node_reference = serialize_node_reference(*output.get_node());
    if (index == 0)
    {
        result = json_node_reference;
    }
    else
    {
        result["node"] = json_node_reference;
        result["index"] = index;
    }
    return result;
}

json JSONSerializer::serialize_output_vector(const OutputVector& output_vector)
{
    json result;
    for (const Output<Node>& output : output_vector)
    {
        result.push_back(serialize_output(output));
    }
    return result;
}

json JSONSerializer::serialize_node(const Node& n)
{
    m_nodes_serialized.insert(&n);
    json node;
    node["name"] = n.get_name();
    auto op_version = n.get_version();
    node["op_version"] = op_version;

    if (n.get_name() != n.get_friendly_name())
    {
        node["friendly_name"] = n.get_friendly_name();
    }
    node["op"] = n.description();
    // TODO Multiple outputs
    json inputs = json::array();
    json control_deps = json::array();
    json outputs = json::array();

    for (auto& input : n.inputs())
    {
        inputs.push_back(serialize_output(input.get_source_output()));
    }
    for (auto cdep : n.get_control_dependencies())
    {
        control_deps.push_back(serialize_node_reference(*cdep));
    }
    for (auto& output : n.outputs())
    {
        outputs.push_back(output.get_tensor().get_name());
    }

    if (!inputs.empty())
    {
        node["inputs"] = inputs;
    }
    if (!control_deps.empty())
    {
        node["control_deps"] = control_deps;
    }
    if (!outputs.empty())
    {
        node["outputs"] = outputs;
    }

    if (s_serialize_output_shapes_enabled)
    {
        json output_shapes = json::array();
        for (size_t i = 0; i < n.get_output_size(); ++i)
        {
            output_shapes.push_back(n.get_output_shape(i));
        }
        node["output_shapes"] = output_shapes;
    }
    if (ngraph::get_provenance_enabled())
    {
        json provenance_tags = json::array();
        for (auto prov_tag : n.get_provenance_tags())
        {
            provenance_tags.push_back(prov_tag);
        }
        node["provenance_tags"] = provenance_tags;
    }

    string node_op = n.description();
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
// #pragma GCC diagnostic error "-Wimplicit-fallthrough"
#endif
    switch (get_typeid(node_op))
    {
    case OP_TYPEID::Abs: { break;
    }
    case OP_TYPEID::Acos: { break;
    }
    case OP_TYPEID::Add:
    {
        auto tmp = dynamic_cast<const op::Add*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
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
    case OP_TYPEID::All:
    {
        auto tmp = dynamic_cast<const op::All*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::AllReduce: { break;
    }
    case OP_TYPEID::And:
    {
        auto tmp = dynamic_cast<const op::And*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Any:
    {
        auto tmp = dynamic_cast<const op::Any*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::Asin: { break;
    }
    case OP_TYPEID::Atan: { break;
    }
    case OP_TYPEID::AvgPool:
    {
        if (op_version == 0)
        {
            auto tmp = dynamic_cast<const op::v0::AvgPool*>(&n);
            node["window_shape"] = tmp->get_window_shape();
            node["window_movement_strides"] = tmp->get_window_movement_strides();
            node["padding_below"] = tmp->get_padding_below();
            node["padding_above"] = tmp->get_padding_above();
            node["include_padding_in_avg_computation"] =
                tmp->get_include_padding_in_avg_computation();
            node["pad_type"] = tmp->get_pad_type();
            if (tmp->get_ceil_mode())
            {
                node["ceil_mode"] = tmp->get_ceil_mode();
            }
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::AvgPool*>(&n);
            node["kernel"] = tmp->get_kernel();
            node["strides"] = tmp->get_strides();
            node["pads_begin"] = tmp->get_pads_begin();
            node["pads_end"] = tmp->get_pads_end();
            node["exclude_pad"] = tmp->get_exclude_pad();
            node["auto_pad"] = tmp->get_auto_pad();
            node["rounding_type"] = tmp->get_rounding_type();
        }
        break;
    }
    case OP_TYPEID::AvgPoolBackprop:
    {
        if (op_version == 0)
        {
            auto tmp = dynamic_cast<const op::v0::AvgPoolBackprop*>(&n);
            node["forward_arg_shape"] = tmp->get_forward_arg_shape();
            node["window_shape"] = tmp->get_window_shape();
            node["window_movement_strides"] = tmp->get_window_movement_strides();
            node["padding_below"] = tmp->get_padding_below();
            node["padding_above"] = tmp->get_padding_above();
            node["include_padding_in_avg_computation"] =
                tmp->get_include_padding_in_avg_computation();
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::AvgPoolBackprop*>(&n);
            node["forward_arg_shape"] = tmp->get_forward_arg_shape();
            node["kernel"] = tmp->get_kernel();
            node["strides"] = tmp->get_strides();
            node["pads_begin"] = tmp->get_pads_begin();
            node["pads_end"] = tmp->get_pads_end();
            node["exclude_pad"] = tmp->get_exclude_pad();
        }
        break;
    }
    case OP_TYPEID::BatchMatMul: { break;
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
        node["axes"] = serialize_axis_set(tmp->get_broadcast_axes());
        node["shape"] = tmp->get_broadcast_shape();
        break;
    }
    case OP_TYPEID::BroadcastDistributed: { break;
    }
    case OP_TYPEID::BroadcastLike:
    {
        auto tmp = dynamic_cast<const op::BroadcastLike*>(&n);
        node["initial_axes"] = serialize_axis_set(tmp->get_initial_broadcast_axes());
        break;
    }
    case OP_TYPEID::Ceiling: { break;
    }
    case OP_TYPEID::Clamp:
    {
        auto tmp = dynamic_cast<const op::Clamp*>(&n);
        node["min"] = tmp->get_min();
        node["max"] = tmp->get_max();
        break;
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
        if (tmp->are_all_data_elements_bitwise_identical() && shape_size(tmp->get_shape()) > 0)
        {
            vector<string> vs;
            vs.push_back(tmp->convert_value_to_string(0));
            node["value"] = vs;
        }
        else
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
        node["pad_type"] = tmp->get_pad_type();
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
    case OP_TYPEID::ConvolutionBias:
    {
        auto tmp = dynamic_cast<const op::ConvolutionBias*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        break;
    }
    case OP_TYPEID::ConvolutionBiasAdd:
    {
        auto tmp = dynamic_cast<const op::ConvolutionBiasAdd*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        break;
    }
    case OP_TYPEID::ConvolutionBiasBackpropFiltersBias:
    {
        auto tmp = dynamic_cast<const op::ConvolutionBiasBackpropFiltersBias*>(&n);
        node["filters_shape"] = tmp->get_filters_shape();
        node["bias_shape"] = tmp->get_bias_shape();
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
        node["axes"] = serialize_axis_set(tmp->get_axes());
        break;
    }
    case OP_TYPEID::DepthToSpace:
    {
        auto tmp = dynamic_cast<const op::DepthToSpace*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["block_size"] = tmp->get_block_size();
        break;
    }
    case OP_TYPEID::Divide:
    {
        auto tmp = dynamic_cast<const op::Divide*>(&n);
        node["pythondiv"] = tmp->is_pythondiv();
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Dot:
    {
        auto tmp = dynamic_cast<const op::Dot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
        break;
    }
    case OP_TYPEID::DynBroadcast: { break;
    }
    case OP_TYPEID::DynPad: { break;
    }
    case OP_TYPEID::DynReplaceSlice:
    {
        auto tmp = dynamic_cast<const op::DynReplaceSlice*>(&n);
        node["lower_bounds_mask"] = tmp->get_lower_bounds_mask();
        node["upper_bounds_mask"] = tmp->get_upper_bounds_mask();
        node["new_axis"] = tmp->get_new_axis();
        node["shrink_axis"] = tmp->get_shrink_axis();
        node["ellipsis_mask"] = tmp->get_ellipsis_mask();
        break;
    }
    case OP_TYPEID::DynReshape: { break;
    }
    case OP_TYPEID::DynSlice:
    {
        auto tmp = dynamic_cast<const op::DynSlice*>(&n);
        node["lower_bounds_mask"] = tmp->get_lower_bounds_mask();
        node["upper_bounds_mask"] = tmp->get_upper_bounds_mask();
        node["new_axis"] = tmp->get_new_axis();
        node["shrink_axis"] = tmp->get_shrink_axis();
        node["ellipsis_mask"] = tmp->get_ellipsis_mask();
        break;
    }
    case OP_TYPEID::Elu:
    {
        auto tmp = dynamic_cast<const op::Elu*>(&n);
        node["alpha"] = tmp->get_alpha();
        break;
    }
    case OP_TYPEID::EmbeddingLookup: { break;
    }
    case OP_TYPEID::Equal:
    {
        auto tmp = dynamic_cast<const op::Equal*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Erf: { break;
    }
    case OP_TYPEID::Exp: { break;
    }
    case OP_TYPEID::FakeQuantize:
    {
        auto tmp = dynamic_cast<const op::FakeQuantize*>(&n);
        node["levels"] = tmp->get_levels();
        break;
    }
    case OP_TYPEID::Floor: { break;
    }
    case OP_TYPEID::Gather:
    {
        if (op_version == 0)
        {
            auto tmp = dynamic_cast<const op::v0::Gather*>(&n);
            node["axis"] = tmp->get_axis();
        }
        break;
    }
    case OP_TYPEID::GatherND: { break;
    }
    case OP_TYPEID::GetOutputElement:
    {
        auto tmp = dynamic_cast<const op::GetOutputElement*>(&n);
        node["n"] = tmp->get_n();
        break;
    }
    case OP_TYPEID::Gelu: { break;
    }
    case OP_TYPEID::GeluBackpropFactor: { break;
    }
    case OP_TYPEID::Gemm:
    {
        auto tmp = dynamic_cast<const op::Gemm*>(&n);
        node["alpha"] = tmp->get_alpha();
        node["beta"] = tmp->get_beta();
        node["transA"] = tmp->get_transA();
        node["transB"] = tmp->get_transB();
        break;
    }
    case OP_TYPEID::GenerateMask:
    {
        auto tmp = dynamic_cast<const op::GenerateMask*>(&n);
        node["output_shape"] = tmp->get_mask_shape();
        node["type"] = write_element_type(tmp->get_element_type());
        node["use_seed"] = tmp->get_use_seed();
        node["seed"] = tmp->get_seed();
        node["probability"] = tmp->get_probability();
        break;
    }
    case OP_TYPEID::Greater:
    {
        auto tmp = dynamic_cast<const op::Greater*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::GreaterEq:
    {
        auto tmp = dynamic_cast<const op::GreaterEq*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::GRN:
    {
        auto tmp = dynamic_cast<const op::GRN*>(&n);
        node["bias"] = tmp->get_bias();
        break;
    }
    case OP_TYPEID::GRUCell:
    {
        auto tmp = dynamic_cast<const op::GRUCell*>(&n);
        node["hidden_size"] = tmp->get_hidden_size();
        node["clip"] = tmp->get_clip();
        node["activations"] = tmp->get_activations();
        node["activation_alpha"] = tmp->get_activation_alpha();
        node["activation_beta"] = tmp->get_activation_beta();
        node["linear_before_reset"] = tmp->get_linear_before_reset();
        break;
    }
    case OP_TYPEID::GroupConvolution:
    {
        auto tmp = dynamic_cast<const op::GroupConvolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        node["groups"] = tmp->get_groups();
        node["pad_type"] = tmp->get_pad_type();
        break;
    }
    case OP_TYPEID::GroupConvolutionTranspose:
    {
        auto tmp = dynamic_cast<const op::GroupConvolutionTranspose*>(&n);
        node["strides"] = tmp->get_strides();
        node["dilations"] = tmp->get_dilations();
        node["padding_begin"] = tmp->get_padding_begin();
        node["padding_end"] = tmp->get_padding_end();
        node["output_padding"] = tmp->get_output_padding();
        node["groups"] = tmp->get_groups();
        node["pad_type"] = tmp->get_pad_type();
        node["output_shape"] = tmp->get_output_shape();
        break;
    }
    case OP_TYPEID::HardSigmoid:
    {
        auto tmp = dynamic_cast<const op::HardSigmoid*>(&n);
        node["alpha"] = tmp->get_alpha();
        node["beta"] = tmp->get_beta();
        break;
    }
    case OP_TYPEID::Less:
    {
        auto tmp = dynamic_cast<const op::Less*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::LessEq:
    {
        auto tmp = dynamic_cast<const op::LessEq*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
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
    case OP_TYPEID::LSTMCell:
    {
        auto tmp = dynamic_cast<const op::LSTMCell*>(&n);
        node["hidden_size"] = tmp->get_hidden_size();
        node["clip"] = tmp->get_clip();
        node["activations"] = tmp->get_activations();
        node["activation_alpha"] = tmp->get_activation_alpha();
        node["activation_beta"] = tmp->get_activation_beta();
        node["input_forget"] = tmp->get_input_forget();
        break;
    }
    case OP_TYPEID::MatMul:
    {
        auto tmp = dynamic_cast<const op::MatMul*>(&n);
        node["transpose_a"] = tmp->get_transpose_a();
        node["transpose_b"] = tmp->get_transpose_b();
        break;
    }
    case OP_TYPEID::Max:
    {
        auto tmp = dynamic_cast<const op::Max*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::MaxPool:
    {
        if (op_version == 0)
        {
            auto tmp = dynamic_cast<const op::v0::MaxPool*>(&n);
            node["window_shape"] = tmp->get_window_shape();
            node["window_movement_strides"] = tmp->get_window_movement_strides();
            node["padding_below"] = tmp->get_padding_below();
            node["padding_above"] = tmp->get_padding_above();
            node["pad_type"] = tmp->get_pad_type();
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::MaxPool*>(&n);
            node["kernel"] = tmp->get_kernel();
            node["strides"] = tmp->get_strides();
            node["pads_begin"] = tmp->get_pads_begin();
            node["pads_end"] = tmp->get_pads_end();
            node["auto_pad"] = tmp->get_auto_pad();
            node["rounding_type"] = tmp->get_rounding_type();
        }
        break;
    }
    case OP_TYPEID::MaxPoolBackprop:
    {
        if (op_version == 0)
        {
            auto tmp = dynamic_cast<const op::v0::MaxPoolBackprop*>(&n);
            node["window_shape"] = tmp->get_window_shape();
            node["window_movement_strides"] = tmp->get_window_movement_strides();
            node["padding_below"] = tmp->get_padding_below();
            node["padding_above"] = tmp->get_padding_above();
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::MaxPoolBackprop*>(&n);
            node["kernel"] = tmp->get_kernel();
            node["strides"] = tmp->get_strides();
            node["pads_begin"] = tmp->get_pads_begin();
            node["pads_end"] = tmp->get_pads_end();
        }
        break;
    }
    case OP_TYPEID::Maximum:
    {
        auto tmp = dynamic_cast<const op::Maximum*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Min:
    {
        auto tmp = dynamic_cast<const op::Min*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::Minimum:
    {
        auto tmp = dynamic_cast<const op::Minimum*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Multiply:
    {
        auto tmp = dynamic_cast<const op::Multiply*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::MVN:
    {
        auto tmp = dynamic_cast<const op::MVN*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        node["normalize_variance"] = tmp->get_normalize_variance();
        node["eps"] = tmp->get_eps();
        break;
    }
    case OP_TYPEID::Negative: { break;
    }
    case OP_TYPEID::NormalizeL2:
    {
        auto tmp = dynamic_cast<const op::NormalizeL2*>(&n);
        node["eps"] = tmp->get_eps();
        node["eps_mode"] = tmp->get_eps_mode();
        break;
    }
    case OP_TYPEID::NotEqual:
    {
        auto tmp = dynamic_cast<const op::NotEqual*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
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
    case OP_TYPEID::Or:
    {
        auto tmp = dynamic_cast<const op::Or*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Pad:
    {
        if (op_version == 0)
        {
            auto tmp = dynamic_cast<const op::v0::Pad*>(&n);
            node["padding_below"] = tmp->get_padding_below();
            node["padding_above"] = tmp->get_padding_above();
            node["pad_mode"] = tmp->get_pad_mode();
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::Pad*>(&n);
            node["pad_mode"] = tmp->get_pad_mode();
        }
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
    case OP_TYPEID::Passthrough:
    {
        auto tmp = dynamic_cast<const op::Passthrough*>(&n);
        node["logical_type"] = tmp->logical_type();
        node["language"] = tmp->language();
        node["function"] = tmp->function();
        std::vector<json> outputs_js;
        for (const auto& output_shape : tmp->output_shapes())
        {
            json output_js;
            output_js["element_type"] = write_element_type(std::get<0>(output_shape));
            output_js["shape"] = write_partial_shape(std::get<1>(output_shape));
            outputs_js.emplace_back(std::move(output_js));
        }
        node["output_shapes"] = std::move(outputs_js);
        break;
    }
    case OP_TYPEID::PRelu: { break;
    }
    case OP_TYPEID::Product:
    {
        if (op_version == 0)
        {
            break;
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::ReduceProd*>(&n);
            node["keep_dims"] = tmp->get_keep_dims();
        }
        break;
    }
    case OP_TYPEID::Power:
    {
        auto tmp = dynamic_cast<const op::Power*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Quantize:
    {
        auto tmp = dynamic_cast<const op::Quantize*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["axes"] = serialize_axis_set(tmp->get_axes());
        node["round_mode"] = tmp->get_round_mode();
        break;
    }
    case OP_TYPEID::QuantizedConvolutionBias: { break;
    }
    case OP_TYPEID::QuantizedConvolutionBiasAdd: { break;
    }
    case OP_TYPEID::QuantizedConvolutionBiasSignedAdd: { break;
    }
    case OP_TYPEID::QuantizedConvolutionRelu: { break;
    }
    case OP_TYPEID::QuantizedConvolution:
    {
        auto tmp = dynamic_cast<const op::QuantizedConvolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        node["output_type"] = write_element_type(tmp->get_element_type());
        node["input_axes"] = tmp->get_input_axes();
        node["filter_axes"] = tmp->get_filter_axes();
        node["output_axes"] = tmp->get_output_axes();
        break;
    }
    case OP_TYPEID::QuantizedDotBias: { break;
    }
    case OP_TYPEID::QuantizedDot:
    {
        auto tmp = dynamic_cast<const op::QuantizedDot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
        node["output_type"] = write_element_type(tmp->get_element_type());
        node["input0_axes"] = tmp->get_input0_axes();
        node["input1_axes"] = tmp->get_input1_axes();
        node["output_axes"] = tmp->get_output_axes();
        break;
    }
    case OP_TYPEID::Recv:
    {
        auto tmp = dynamic_cast<const op::Recv*>(&n);
        node["source_id"] = tmp->get_src_id();
        break;
    }
    case OP_TYPEID::RandomUniform:
    {
        auto tmp = dynamic_cast<const op::RandomUniform*>(&n);
        node["fixed_seed"] = tmp->get_fixed_seed();
        break;
    }
    case OP_TYPEID::Range: { break;
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
    case OP_TYPEID::Result:
    {
        auto tmp = dynamic_cast<const op::Result*>(&n);
        node["needs_default_layout"] = tmp->needs_default_layout();
        break;
    }
    case OP_TYPEID::Reverse:
    {
        auto tmp = dynamic_cast<const op::Reverse*>(&n);
        node["reversed_axes"] = serialize_axis_set(tmp->get_reversed_axes());
        break;
    }
    case OP_TYPEID::ReverseSequence:
    {
        auto tmp = dynamic_cast<const op::ReverseSequence*>(&n);
        node["batch_axis"] = tmp->get_batch_axis();
        node["sequence_axis"] = tmp->get_sequence_axis();
        break;
    }
    case OP_TYPEID::RNNCell:
    {
        auto tmp = dynamic_cast<const op::RNNCell*>(&n);
        node["hidden_size"] = tmp->get_hidden_size();
        node["clip"] = tmp->get_clip();
        node["activations"] = tmp->get_activations();
        node["activation_alpha"] = tmp->get_activation_alpha();
        node["activation_beta"] = tmp->get_activation_beta();
        break;
    }
    case OP_TYPEID::ScalarConstantLike:
    {
        auto tmp = dynamic_cast<const op::ScalarConstantLikeBase*>(&n);
        auto constant = tmp->as_constant();
        node["value"] = constant->get_value_strings()[0];
        node["element_type"] = write_element_type(constant->get_element_type());
        break;
    }
    case OP_TYPEID::ScaleShift: { break;
    }
    case OP_TYPEID::ScatterAdd: { break;
    }
    case OP_TYPEID::ScatterNDAdd: { break;
    }
    case OP_TYPEID::Select: { break;
    }
    case OP_TYPEID::Send:
    {
        auto tmp = dynamic_cast<const op::Send*>(&n);
        node["dest_id"] = tmp->get_dest_id();
        break;
    }
    case OP_TYPEID::ShapeOf: { break;
    }
    case OP_TYPEID::ShuffleChannels:
    {
        const auto tmp = dynamic_cast<const op::ShuffleChannels*>(&n);
        node["axis"] = tmp->get_axis();
        node["groups"] = tmp->get_groups();
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
    case OP_TYPEID::SpaceToDepth:
    {
        auto tmp = dynamic_cast<const op::SpaceToDepth*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["block_size"] = tmp->get_block_size();
        break;
    }
    case OP_TYPEID::Split:
    {
        auto tmp = dynamic_cast<const op::Split*>(&n);
        node["axis"] = tmp->get_axis();
        node["splits"] = tmp->get_splits();
        break;
    }
    case OP_TYPEID::Sqrt: { break;
    }
    case OP_TYPEID::SquaredDifference: { break;
    }
    case OP_TYPEID::Squeeze: { break;
    }
    case OP_TYPEID::StopGradient: { break;
    }
    case OP_TYPEID::Subtract:
    {
        auto tmp = dynamic_cast<const op::Subtract*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Sum:
    {
        if (op_version == 0)
        {
            break;
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::ReduceSum*>(&n);
            node["keep_dims"] = tmp->get_keep_dims();
        }
        break;
    }
    case OP_TYPEID::Softmax:
    {
        if (op_version == 0)
        {
            auto tmp = dynamic_cast<const op::v0::Softmax*>(&n);
            node["softmax_axes"] = serialize_axis_set(tmp->get_axes());
        }
        if (op_version == 1)
        {
            auto tmp = dynamic_cast<const op::v1::Softmax*>(&n);
            node["softmax_axis"] = tmp->get_axis();
        }
        break;
    }
    case OP_TYPEID::Tan: { break;
    }
    case OP_TYPEID::Tanh: { break;
    }
    case OP_TYPEID::Tile: { break;
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
    case OP_TYPEID::Transpose: { break;
    }
    case OP_TYPEID::Unsqueeze: { break;
    }
    case OP_TYPEID::Xor:
    {
        auto tmp = dynamic_cast<const op::Xor*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::UnknownOp: { break;
    }
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
    return node;
}
