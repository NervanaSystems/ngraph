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

#include <fstream>
#include <functional>
#include <queue>
#include <stack>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/cpio.hpp"
#include "ngraph/env_util.hpp"
#include "ngraph/factory.hpp"
#include "ngraph/file_util.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ops.hpp"
#include "ngraph/provenance.hpp"
#include "ngraph/serializer.hpp"
#include "ngraph/util.hpp"
#include "nlohmann/json.hpp"

using namespace ngraph;
using namespace std;
using json = nlohmann::json;
using const_data_callback_t = shared_ptr<Node>(const string&, const element::Type&, const Shape&);

static json write_element_type(const ngraph::element::Type& n);
static element::Type read_element_type(json j);
static json write_partial_shape(const PartialShape& s);
static PartialShape read_partial_shape(json j);

namespace
{
#define OBSOLETE_OPS                                                                               \
    NGRAPH_OP(Add, 0)                                                                              \
    NGRAPH_OP(And, 0)                                                                              \
    NGRAPH_OP(Divide, 0)                                                                           \
    NGRAPH_OP(Equal, 0)                                                                            \
    NGRAPH_OP(GetOutputElement, 0)                                                                 \
    NGRAPH_OP(Greater, 0)                                                                          \
    NGRAPH_OP(GreaterEq, 0)                                                                        \
    NGRAPH_OP(Less, 0)                                                                             \
    NGRAPH_OP(LessEq, 0)                                                                           \
    NGRAPH_OP(Maximum, 0)                                                                          \
    NGRAPH_OP(Minimum, 0)                                                                          \
    NGRAPH_OP(Multiply, 0)                                                                         \
    NGRAPH_OP(Not, 0)                                                                              \
    NGRAPH_OP(NotEqual, 0)                                                                         \
    NGRAPH_OP(Or, 0)                                                                               \
    NGRAPH_OP(Power, 0)                                                                            \
    NGRAPH_OP(Subtract, 0)                                                                         \
    NGRAPH_OP(Xor, 0)

    // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
    // Abs,
    // Acos,
    // ...
    enum class OP_TYPEID
    {
#define NGRAPH_OP(NAME, VERSION) NAME##_v##VERSION,
#include "ngraph/op_version_tbl.hpp"
        OBSOLETE_OPS
#undef NGRAPH_OP
            UnknownOp
    };
}

static OP_TYPEID get_typeid(const string& type_info)
{
    // This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
    // {"Abs_v0", OP_TYPEID::Abs},
    // {"Acos_v0", OP_TYPEID::Acos},
    // ...
    static const map<string, OP_TYPEID> type_info_map{
#define NGRAPH_OP(NAME, VERSION) {#NAME "_v" #VERSION, OP_TYPEID::NAME##_v##VERSION},
#include "ngraph/op_version_tbl.hpp"
        // Still need to deserialize GetOutputElement because it may be in some old json files
        // This is just to handle such cases.
        OBSOLETE_OPS
#undef NGRAPH_OP
    };
    OP_TYPEID rc = OP_TYPEID::UnknownOp;

    auto it = type_info_map.find(type_info);
    if (it != type_info_map.end())
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

class JSONAttributeSerializer : public AttributeVisitor
{
public:
    JSONAttributeSerializer(json& j)
        : m_json(j)
    {
    }
    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
    {
        NGRAPH_CHECK(false, "Adapter ", adapter.get_type_info().name, " is not handled");
    }
    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter) override
    {
        m_json[name] = adapter.get();
    }
    void on_adapter(const std::string& name,
                    ValueAccessor<std::vector<std::string>>& adapter) override
    {
        m_json[name] = adapter.get();
    }

protected:
    json& m_json;
};

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
    json serialize_node(const Node& node);
    json serialize_axis_set(const AxisSet& axis_set);
    json serialize_tensor_iterator_input_description(
        const std::shared_ptr<op::v0::TensorIterator::InputDescription>&);
    json serialize_tensor_iterator_output_description(
        const std::shared_ptr<op::v0::TensorIterator::OutputDescription>&);

protected:
    size_t m_indent{0};
    bool m_serialize_output_shapes{false};
    bool m_binary_constant_data{false};
    json m_json_nodes;
};

class JSONAttributeDeserializer : public AttributeVisitor
{
public:
    JSONAttributeDeserializer(json& j)
        : m_json(j)
    {
    }
    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override
    {
        NGRAPH_CHECK(false, "Adapter ", adapter.get_type_info().name, " is not handled");
    }
    void on_adapter(const std::string& name, ValueAccessor<std::string>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::string>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<bool>());
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<int64_t>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<int64_t>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<double>());
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<std::vector<int64_t>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<int64_t>>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<uint64_t>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<uint64_t>>());
        }
    }
    void on_adapter(const std::string& name, ValueAccessor<std::vector<float>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<float>>());
        }
    }
    void on_adapter(const std::string& name,
                    ValueAccessor<std::vector<std::string>>& adapter) override
    {
        if (has_key(m_json, name))
        {
            adapter.set(m_json.at(name).get<std::vector<std::string>>());
        }
    }

protected:
    json& m_json;
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
    shared_ptr<op::v0::TensorIterator::InputDescription>
        deserialize_tensor_iterator_input_description(json j);
    shared_ptr<op::v0::TensorIterator::OutputDescription>
        deserialize_tensor_iterator_output_description(json j);

protected:
    unordered_map<string, shared_ptr<Node>> m_node_map;
    unordered_map<string, shared_ptr<Function>> m_function_map;
    function<const_data_callback_t> m_const_data_callback;
    map<string, Output<Node>> m_goe_alias;
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
        return d.get_length();
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
        return Dimension(static_cast<int64_t>(j));
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
        std::vector<json> vals(s.rank().get_length());
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

static op::AutoBroadcastSpec
    read_auto_broadcast(json js_node,
                        const std::string& attr,
                        const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec())
{
    if (has_key(js_node, attr))
    {
        json j = js_node[attr];
        return op::AutoBroadcastSpec(static_cast<op::AutoBroadcastType>(j.at("type")),
                                     j.at("axis").get<int64_t>());
    }
    else
    {
        return autob;
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

static json write_element_type(const ngraph::element::Type& n)
{
    json j;
    j = n.c_type_string();
    return j;
}

static element::Type read_element_type(json j)
{
    string c_type_string;
    element::Type rc;
    if (j.is_object())
    {
        c_type_string = j.at("c_type_string").get<string>();
    }
    else
    {
        c_type_string = j.get<string>();
    }
    for (element::Type t : element::Type::get_known_types())
    {
        if (t.c_type_string() == c_type_string)
        {
            rc = t;
            break;
        }
    }
    return rc;
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
                       if (auto c = node->as_type<op::v0::Constant>())
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
                            const_node = make_shared<op::v0::Constant>(et, shape, const_data);
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
        json_parameters.push_back(param->get_name());
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
        function["result"].push_back(f.get_output_op(i)->get_name());
    }

    json nodes;
    for (shared_ptr<Node> node : f.get_ordered_ops())
    {
        nodes.push_back(serialize_node(*node));
    }

    function["ops"] = nodes;
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
    const string& name = json_node_reference;
    auto it = m_goe_alias.find(name);
    if (it != m_goe_alias.end())
    {
        return it->second.get_node_shared_ptr();
    }
    else
    {
        return Output<Node>(m_node_map.at(name), index);
    }
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

json JSONSerializer::serialize_tensor_iterator_input_description(
    const std::shared_ptr<op::v0::TensorIterator::InputDescription>& input_description)
{
    json result;
    if (auto slice = as_type_ptr<op::v0::TensorIterator::SliceInputDescription>(input_description))
    {
        result["kind"] = "slice";
        result["input_index"] = slice->m_input_index;
        result["body_parameter_index"] = slice->m_body_parameter_index;
        result["start"] = slice->m_start;
        result["stride"] = slice->m_stride;
        result["part_size"] = slice->m_part_size;
        result["end"] = slice->m_end;
        result["axis"] = slice->m_axis;
    }
    else if (auto merged =
                 as_type_ptr<op::v0::TensorIterator::MergedInputDescription>(input_description))
    {
        result["kind"] = "merged";
        result["input_index"] = merged->m_input_index;
        result["body_parameter_index"] = merged->m_body_parameter_index;
        result["body_value_index"] = merged->m_body_value_index;
    }
    else if (auto constant =
                 as_type_ptr<op::v0::TensorIterator::InvariantInputDescription>(input_description))
    {
        result["kind"] = "constant";
        result["input_index"] = constant->m_input_index;
        result["body_parameter_index"] = constant->m_body_parameter_index;
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type");
    }
    return result;
}

shared_ptr<op::v0::TensorIterator::InputDescription>
    JSONDeserializer::deserialize_tensor_iterator_input_description(json j)
{
    string kind = j["kind"];
    shared_ptr<op::v0::TensorIterator::InputDescription> result;
    if (kind == "slice")
    {
        uint64_t input_index = j["input_index"].get<uint64_t>();
        uint64_t body_parameter_index = j["body_parameter_index"].get<uint64_t>();
        int64_t start = j["start"].get<int64_t>();
        int64_t stride = j["stride"].get<int64_t>();
        uint64_t part_size = j["part_size"].get<int64_t>();
        int64_t end = j["end"].get<int64_t>();
        int64_t axis = j["axis"].get<int64_t>();
        result = make_shared<op::v0::TensorIterator::SliceInputDescription>(
            input_index, body_parameter_index, start, stride, part_size, end, axis);
    }
    else if (kind == "merged")
    {
        uint64_t input_index = j["input_index"].get<uint64_t>();
        uint64_t body_parameter_index = j["body_parameter_index"].get<uint64_t>();
        uint64_t body_value_index = j["body_value_index"].get<uint64_t>();
        result = make_shared<op::v0::TensorIterator::MergedInputDescription>(
            input_index, body_parameter_index, body_value_index);
    }
    else if (kind == "constant")
    {
        uint64_t input_index = j["input_index"].get<uint64_t>();
        uint64_t body_parameter_index = j["body_parameter_index"].get<uint64_t>();
        result = make_shared<op::v0::TensorIterator::InvariantInputDescription>(
            input_index, body_parameter_index);
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type: ", kind);
    }
    return result;
}

json JSONSerializer::serialize_tensor_iterator_output_description(
    const std::shared_ptr<op::v0::TensorIterator::OutputDescription>& output_description)
{
    json result;
    if (auto concat =
            as_type_ptr<op::v0::TensorIterator::ConcatOutputDescription>(output_description))
    {
        result["kind"] = "concat";
        result["body_value_index"] = concat->m_body_value_index;
        result["output_index"] = concat->m_output_index;
        result["start"] = concat->m_start;
        result["stride"] = concat->m_stride;
        result["part_size"] = concat->m_part_size;
        result["end"] = concat->m_end;
        result["axis"] = concat->m_axis;
    }
    else if (auto body_output =
                 as_type_ptr<op::v0::TensorIterator::BodyOutputDescription>(output_description))
    {
        result["kind"] = "body_output";
        result["body_value_index"] = body_output->m_body_value_index;
        result["output_index"] = body_output->m_output_index;
        result["iteration"] = body_output->m_iteration;
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type");
    }
    return result;
}

std::shared_ptr<op::v0::TensorIterator::OutputDescription>
    JSONDeserializer::deserialize_tensor_iterator_output_description(json j)
{
    string kind = j["kind"];
    shared_ptr<op::v0::TensorIterator::OutputDescription> result;
    if (kind == "concat")
    {
        uint64_t body_value_index = j["body_value_index"].get<uint64_t>();
        uint64_t output_index = j["output_index"].get<uint64_t>();
        int64_t start = j["start"].get<int64_t>();
        int64_t stride = j["stride"].get<int64_t>();
        uint64_t part_size = j["part_size"].get<int64_t>();
        int64_t end = j["end"].get<int64_t>();
        int64_t axis = j["axis"].get<int64_t>();
        result = make_shared<op::v0::TensorIterator::ConcatOutputDescription>(
            body_value_index, output_index, start, stride, part_size, end, axis);
    }
    else if (kind == "body_output")
    {
        uint64_t body_value_index = j["body_value_index"].get<uint64_t>();
        uint64_t output_index = j["output_index"].get<uint64_t>();
        int64_t iteration = j["iteration"].get<int64_t>();
        result = make_shared<op::v0::TensorIterator::BodyOutputDescription>(
            body_value_index, output_index, iteration);
    }
    else
    {
        NGRAPH_UNREACHABLE("Unknown input description type: ", kind);
    }
    return result;
}

ParameterVector JSONDeserializer::deserialize_parameter_vector(json json_parameters)
{
    ParameterVector params;
    for (auto& param_ref : json_parameters)
    {
        params.push_back(as_type_ptr<op::v0::Parameter>(deserialize_node_reference(param_ref)));
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

    // This handles both graphs w/ `op::v0::Result` and legacy graphs w/o it
    // If we are dealing w/ a legacy graph, add op::v0::Result for each output node
    ResultVector result;
    size_t results = 0;
    for (auto& result_ref : func_result)
    {
        auto fr = deserialize_node_reference(result_ref);
        if (auto res = as_type_ptr<op::v0::Result>(fr))
        {
            result.push_back(res);
            // make sure we have `op::v0::Result` on top of all outputs
            results++;
        }
        else
        {
            result.push_back(std::make_shared<op::v0::Result>(fr));
        }
    }

    if (results != 0 && results != func_result.size())
    {
        throw ngraph_error(
            "Graph serialization is inconsistent. Some op::v0::Results appear to be missing");
    }

    ParameterVector params = deserialize_parameter_vector(func_js.at("parameters"));

    shared_ptr<Function> rc{make_shared<Function>(result, params, func_name)};
    m_function_map[func_name] = rc;
    return rc;
}

shared_ptr<Node> JSONDeserializer::deserialize_node(json node_js)
{
    auto& factory_registry = FactoryRegistry<Node>::get();
    shared_ptr<Node> node;
    try
    {
        string node_op = node_js.at("op").get<string>();
        size_t op_version = get_value<size_t>(node_js, "op_version");
        Node::type_info_t type_info{node_op.c_str(), op_version};
        string node_name = node_js.at("name").get<string>();
        string friendly_name = get_value<string>(node_js, "friendly_name");
        vector<json> control_deps_inputs = get_value<vector<json>>(node_js, "control_deps");
        vector<string> node_outputs = get_value<vector<string>>(node_js, "outputs");
        OutputVector args(deserialize_output_vector(node_js["inputs"]));
        if (has_key(node_js, "attribute_visitor"))
        {
            if (factory_registry.has_factory(type_info))
            {
                node = shared_ptr<Node>(factory_registry.create(type_info));
                JSONAttributeDeserializer visitor(node_js);
                node->set_arguments(static_cast<OutputVector>(args));
                node->visit_attributes(visitor);
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
                node->constructor_validate_and_infer_types();
                m_node_map[node_name] = node;
                return node;
            }
        }

        string op_full_name = node_op + "_v" + to_string(op_version);

#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
// #pragma GCC diagnostic error "-Wswitch-enum"
// #pragma GCC diagnostic error "-Wimplicit-fallthrough"
#endif

        switch (get_typeid(op_full_name))
        {
        case OP_TYPEID::Abs_v0:
        {
            node = make_shared<op::v0::Abs>(args[0]);
            break;
        }
        case OP_TYPEID::Acos_v0:
        {
            node = make_shared<op::v0::Acos>(args[0]);
            break;
        }
        case OP_TYPEID::Add_v0:
        {
            node = make_shared<op::v1::Add>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::All_v0:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::v0::All>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::AllReduce_v0:
        {
            node = make_shared<op::v0::AllReduce>(args[0]);
            break;
        }
        case OP_TYPEID::And_v0:
        {
            node = make_shared<op::v1::LogicalAnd>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Any_v0:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::v0::Any>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::ArgMin_v0:
        {
            auto axis = node_js.at("axis").get<size_t>();
            auto target_type = read_element_type(node_js.at("index_element_type"));
            node = make_shared<op::v0::ArgMin>(args[0], axis, target_type);
            break;
        }
        case OP_TYPEID::ArgMax_v0:
        {
            auto axis = node_js.at("axis").get<size_t>();
            auto target_type = read_element_type(node_js.at("index_element_type"));
            node = make_shared<op::v0::ArgMax>(args[0], axis, target_type);
            break;
        }
        case OP_TYPEID::Asin_v0:
        {
            node = make_shared<op::v0::Asin>(args[0]);
            break;
        }
        case OP_TYPEID::Atan_v0:
        {
            node = make_shared<op::v0::Atan>(args[0]);
            break;
        }
        case OP_TYPEID::Atan2_v0:
        {
            node =
                make_shared<op::v0::Atan2>(args[0], args[1], read_auto_broadcast(node_js, "autob"));
            break;
        }

        case OP_TYPEID::AvgPool_v0:
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
            break;
        }
        case OP_TYPEID::AvgPoolBackprop_v0:
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
            break;
        }
        case OP_TYPEID::BatchMatMul_v0:
        {
            node = make_shared<op::v0::BatchMatMul>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::BatchMatMulTranspose_v0:
        {
            auto transpose_0 = node_js.at("transpose_0").get<bool>();
            auto transpose_1 = node_js.at("transpose_1").get<bool>();
            node = make_shared<op::v0::BatchMatMulTranspose>(
                args[0], args[1], transpose_0, transpose_1);
            break;
        }
        case OP_TYPEID::BatchNormTraining_v0:
        {
            auto epsilon = node_js.at("eps").get<double>();
            // Odd order for back-compatibility
            node = make_shared<op::v0::BatchNormTraining>(args[2], args[0], args[1], epsilon);
            break;
        }
        case OP_TYPEID::BatchNormInference_v0:
        {
            auto epsilon = node_js.at("eps").get<double>();
            // Odd order for back-compatibility
            node = make_shared<op::v0::BatchNormInference>(
                args[2], args[0], args[1], args[3], args[4], epsilon);
            break;
        }
        case OP_TYPEID::BatchNormTrainingBackprop_v0:
        {
            auto epsilon = node_js.at("eps").get<double>();
            // Odd order for back-compatibility
            node = make_shared<op::v0::BatchNormTrainingBackprop>(
                args[2], args[0], args[1], args[3], args[4], args[5], epsilon);
            break;
        }
        case OP_TYPEID::Broadcast_v0:
        {
            auto shape = node_js.at("shape").get<vector<size_t>>();
            auto axes = deserialize_axis_set(node_js.at("axes"));
            node = make_shared<op::v0::Broadcast>(args[0], shape, axes);
            break;
        }
        case OP_TYPEID::BroadcastDistributed_v0:
        {
            node = make_shared<op::v0::BroadcastDistributed>(args[0]);
            break;
        }
        case OP_TYPEID::BroadcastLike_v0:
        {
            auto initial_axes = deserialize_axis_set(node_js.at("initial_axes"));
            node = make_shared<op::v0::BroadcastLike>(args[0], args[1], initial_axes);
            break;
        }
        case OP_TYPEID::Ceiling_v0:
        {
            node = make_shared<op::v0::Ceiling>(args[0]);
            break;
        }
        case OP_TYPEID::Clamp_v0:
        {
            const double clamp_min = parse_string<double>(node_js.at("min").get<string>());
            const double clamp_max = parse_string<double>(node_js.at("max").get<string>());
            node = make_shared<op::v0::Clamp>(args[0], clamp_min, clamp_max);
            break;
        }
        case OP_TYPEID::Concat_v0:
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::v0::Concat>(static_cast<OutputVector>(args), axis);
            break;
        }
        case OP_TYPEID::Constant_v0:
        {
            auto type_node_js =
                has_key(node_js, "element_type") ? node_js : node_js.at("value_type");
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            auto value = node_js.at("value").get<vector<string>>();
            node = make_shared<op::v0::Constant>(element_type, shape, value);
            break;
        }
        case OP_TYPEID::Convert_v0:
        {
            auto target_type = read_element_type(node_js.at("target_type"));
            node = make_shared<op::v0::Convert>(args[0], target_type);
            break;
        }
        case OP_TYPEID::Convolution_v0:
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
                node = make_shared<op::v0::Convolution>(args[0],
                                                        args[1],
                                                        window_movement_strides,
                                                        window_dilation_strides,
                                                        padding_below,
                                                        padding_above);
            }
            else
            {
                node = make_shared<op::v0::Convolution>(
                    args[0],
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
        case OP_TYPEID::ConvolutionBackpropData_v0:
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
            node = make_shared<op::v0::ConvolutionBackpropData>(data_batch_shape,
                                                                args[0],
                                                                args[1],
                                                                window_movement_strides_forward,
                                                                window_dilation_strides_forward,
                                                                padding_below_forward,
                                                                padding_above_forward,
                                                                data_dilation_strides_forward);
            break;
        }
        case OP_TYPEID::ConvolutionBackpropFilters_v0:
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
            node = make_shared<op::v0::ConvolutionBackpropFilters>(args[0],
                                                                   filters_shape,
                                                                   args[1],
                                                                   window_movement_strides_forward,
                                                                   window_dilation_strides_forward,
                                                                   padding_below_forward,
                                                                   padding_above_forward,
                                                                   data_dilation_strides_forward);
            break;
        }
        case OP_TYPEID::ConvolutionBias_v0:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();

            node = make_shared<op::v0::ConvolutionBias>(args[0],
                                                        args[1],
                                                        args[2],
                                                        window_movement_strides,
                                                        window_dilation_strides,
                                                        padding_below,
                                                        padding_above,
                                                        data_dilation_strides);
            break;
        }
        case OP_TYPEID::ConvolutionBiasAdd_v0:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();

            node = make_shared<op::v0::ConvolutionBiasAdd>(args[0],
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
        case OP_TYPEID::ConvolutionBiasBackpropFiltersBias_v0:
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
            node = make_shared<op::v0::ConvolutionBiasBackpropFiltersBias>(
                args[0],
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
        case OP_TYPEID::Cos_v0:
        {
            node = make_shared<op::v0::Cos>(args[0]);
            break;
        }
        case OP_TYPEID::Cosh_v0:
        {
            node = make_shared<op::v0::Cosh>(args[0]);
            break;
        }
        case OP_TYPEID::CumSum_v0:
        {
            auto exclusive = node_js.at("exclusive");
            auto reverse = node_js.at("reverse");
            node = make_shared<op::v0::CumSum>(args[0], args[1], exclusive, reverse);
            break;
        }
        case OP_TYPEID::CrossEntropy_v0:
        {
            auto soft_label = node_js.at("soft_label");
            auto ignore_index = node_js.at("ignore_index");
            node = make_shared<op::v0::CrossEntropy>(args[0], args[1], soft_label, ignore_index);
            break;
        }
        case OP_TYPEID::CrossEntropyBackprop_v0:
        {
            auto soft_label = node_js.at("soft_label");
            auto ignore_index = node_js.at("ignore_index");
            node = make_shared<op::v0::CrossEntropyBackprop>(
                args[0], args[1], args[2], soft_label, ignore_index);
            break;
        }
        case OP_TYPEID::CropAndResize_v0:
        {
            auto resize_method = as_type<op::v0::CropAndResize::ResizeMethod>(
                node_js.at("resize_method").get<string>());
            auto extrapolation_value = node_js.at("extrapolation_value").get<float>();
            node = make_shared<op::v0::CropAndResize>(
                args[0], args[1], args[2], args[3], resize_method, extrapolation_value);
            break;
        }
        case OP_TYPEID::CTCGreedyDecoder_v0: { break;
        }
        case OP_TYPEID::DeformableConvolution_v1:
        {
            const auto strides = node_js.at("strides").get<vector<size_t>>();
            const auto dilations = node_js.at("dilations").get<vector<size_t>>();
            const auto pads_begin = node_js.at("pads_begin").get<vector<std::ptrdiff_t>>();
            const auto pads_end = node_js.at("pads_end").get<vector<std::ptrdiff_t>>();
            const auto group = node_js.at("group").get<size_t>();
            const auto deformable_group = node_js.at("deformable_group").get<size_t>();

            const op::PadType auto_pad = read_pad_type(node_js);

            node = make_shared<op::v1::DeformableConvolution>(args[0],
                                                              args[1],
                                                              args[2],
                                                              strides,
                                                              pads_begin,
                                                              pads_end,
                                                              dilations,
                                                              auto_pad,
                                                              group,
                                                              deformable_group);
            break;
        }
        case OP_TYPEID::DepthToSpace_v0:
        {
            auto mode = node_js.at("mode").get<op::v0::DepthToSpace::DepthToSpaceMode>();
            auto block_size = node_js.at("block_size").get<size_t>();
            node = make_shared<op::v0::DepthToSpace>(args[0], mode, block_size);
            break;
        }
        case OP_TYPEID::DetectionOutput_v0: { break;
        }
        case OP_TYPEID::Dequantize_v0:
        {
            auto type = read_element_type(node_js.at("type"));
            auto axes = deserialize_axis_set(node_js.at("axes"));
            node = make_shared<op::v0::Dequantize>(args[0], args[1], args[2], type, axes);
            break;
        }
        case OP_TYPEID::Divide_v0:
        {
            bool pythondiv = get_or_default(node_js, "pythondiv", true);
            node = make_shared<op::v1::Divide>(
                args[0], args[1], pythondiv, read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Dot_v0:
        {
            // For backwards compatibility, reduction_axes_count is optional.
            if (has_key(node_js, "reduction_axes_count"))
            {
                size_t reduction_axes_count = node_js["reduction_axes_count"].get<size_t>();
                node = make_shared<op::v0::Dot>(args[0], args[1], reduction_axes_count);
            }
            else
            {
                node = make_shared<op::v0::Dot>(args[0], args[1]);
            }
            break;
        }
        case OP_TYPEID::DynBroadcast_v0:
        {
            node = make_shared<op::v0::DynBroadcast>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::DynPad_v0:
        {
            node = make_shared<op::v0::DynPad>(args[0], args[1], args[2], args[3]);
            break;
        }
        case OP_TYPEID::DynReplaceSlice_v0:
        {
            auto lower_bounds_mask = node_js.at("lower_bounds_mask").get<set<size_t>>();
            auto upper_bounds_mask = node_js.at("upper_bounds_mask").get<set<size_t>>();
            auto new_axis = node_js.at("new_axis").get<set<size_t>>();
            auto shrink_axis = node_js.at("shrink_axis").get<set<size_t>>();
            auto ellipsis_mask = node_js.at("ellipsis_mask").get<set<size_t>>();
            node = make_shared<op::v0::DynReplaceSlice>(args[0],
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
        case OP_TYPEID::DynSlice_v0:
        {
            auto lower_bounds_mask = node_js.at("lower_bounds_mask").get<set<size_t>>();
            auto upper_bounds_mask = node_js.at("upper_bounds_mask").get<set<size_t>>();
            auto new_axis = node_js.at("new_axis").get<set<size_t>>();
            auto shrink_axis = node_js.at("shrink_axis").get<set<size_t>>();
            auto ellipsis_mask = node_js.at("ellipsis_mask").get<set<size_t>>();
            node = make_shared<op::v0::DynSlice>(args[0],
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
        case OP_TYPEID::Elu_v0:
        {
            auto alpha = node_js.at("alpha").get<double>();
            node = make_shared<op::v0::Elu>(args[0], alpha);
            break;
        }
        case OP_TYPEID::EmbeddingLookup_v0:
        {
            node = make_shared<op::v0::EmbeddingLookup>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Equal_v0:
        {
            node = make_shared<op::v1::Equal>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Equal_v1:
        {
            node = make_shared<op::v1::Equal>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Erf_v0:
        {
            node = make_shared<op::v0::Erf>(args[0]);
            break;
        }
        case OP_TYPEID::Exp_v0:
        {
            node = make_shared<op::v0::Exp>(args[0]);
            break;
        }
        case OP_TYPEID::FakeQuantize_v0:
        {
            size_t levels = node_js.at("levels").get<size_t>();
            node = make_shared<op::v0::FakeQuantize>(
                args[0], args[1], args[2], args[3], args[4], levels);
            break;
        }
        case OP_TYPEID::Floor_v0:
        {
            node = make_shared<op::v0::Floor>(args[0]);
            break;
        }
        case OP_TYPEID::Gather_v0:
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::v0::Gather>(args[0], args[1], axis);
            break;
        }
        case OP_TYPEID::GatherND_v0:
        {
            node = make_shared<op::v0::GatherND>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Gelu_v0:
        {
            node = make_shared<op::v0::Gelu>(args[0]);
            break;
        }
        case OP_TYPEID::GeluBackpropFactor_v0:
        {
            node = make_shared<op::v0::GeluBackpropFactor>(args[0]);
            break;
        }
        case OP_TYPEID::Gemm_v0:
        {
            auto alpha = node_js.at("alpha").get<double>();
            auto beta = node_js.at("beta").get<double>();
            auto transA = node_js.at("transA").get<bool>();
            auto transB = node_js.at("transB").get<bool>();
            node =
                make_shared<op::v0::Gemm>(args[0], args[1], args[2], alpha, beta, transA, transB);
            break;
        }
        case OP_TYPEID::GenerateMask_v0:
        {
            auto type = read_element_type(node_js.at("type"));
            auto seed = node_js.at("seed").get<unsigned int>();
            auto probability = node_js.at("probability").get<double>();
            bool use_seed = get_or_default<bool>(node_js, "use_seed", false);

            auto output_shape = node_js.at("output_shape").get<vector<size_t>>();

            node = make_shared<op::v0::GenerateMask>(
                args[0], output_shape, type, seed, probability, use_seed);

            break;
        }
        case OP_TYPEID::GetOutputElement_v0:
        {
            m_goe_alias.insert({node_name, args[0]});
            break;
        }
        case OP_TYPEID::Greater_v0:
        {
            node = make_shared<op::v1::Greater>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Greater_v1:
        {
            node = make_shared<op::v1::Greater>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::GreaterEq_v0:
        {
            node = make_shared<op::v1::GreaterEqual>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::GRN_v0:
        {
            auto bias = node_js.at("bias").get<float>();
            node = make_shared<op::v0::GRN>(args[0], bias);
            break;
        }
        case OP_TYPEID::GroupConvolution_v0:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto data_dilation_strides = node_js.at("data_dilation_strides").get<vector<size_t>>();
            op::PadType pad_type = read_pad_type(node_js);
            if (has_key(node_js, "groups"))
            {
                auto groups = node_js.at("groups").get<size_t>();
                node = make_shared<op::v0::GroupConvolution>(args[0],
                                                             args[1],
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             groups,
                                                             pad_type);
            }
            else
            {
                node = make_shared<op::v0::GroupConvolution>(args[0],
                                                             args[1],
                                                             window_movement_strides,
                                                             window_dilation_strides,
                                                             padding_below,
                                                             padding_above,
                                                             data_dilation_strides,
                                                             pad_type);
            }
            break;
        }
        case OP_TYPEID::GroupConvolutionBackpropData_v0:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto groups = node_js.at("groups").get<size_t>();

            node = make_shared<op::v0::GroupConvolutionBackpropData>(args[0],
                                                                     args[1],
                                                                     args[2],
                                                                     window_movement_strides,
                                                                     window_dilation_strides,
                                                                     padding_below,
                                                                     padding_above,
                                                                     groups);
            break;
        }
        case OP_TYPEID::GroupConvolutionBackpropFilters_v0:
        {
            auto window_movement_strides =
                node_js.at("window_movement_strides").get<vector<size_t>>();
            auto window_dilation_strides =
                node_js.at("window_dilation_strides").get<vector<size_t>>();
            auto padding_below = node_js.at("padding_below").get<vector<std::ptrdiff_t>>();
            auto padding_above = node_js.at("padding_above").get<vector<std::ptrdiff_t>>();
            auto groups = node_js.at("groups").get<size_t>();

            node = make_shared<op::v0::GroupConvolutionBackpropFilters>(args[0],
                                                                        args[1],
                                                                        args[2],
                                                                        window_movement_strides,
                                                                        window_dilation_strides,
                                                                        padding_below,
                                                                        padding_above,
                                                                        groups);
            break;
        }
        case OP_TYPEID::HardSigmoid_v0:
        {
            node = make_shared<op::v0::HardSigmoid>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::LayerNorm_v0:
        {
            auto keep_stats = node_js.at("keep_stats").get<bool>();
            auto use_affine = node_js.at("use_affine").get<bool>();
            auto epsilon = node_js.at("epsilon").get<double>();
            auto begin_norm_axis = node_js.at("begin_norm_axis").get<int64_t>();
            if (use_affine)
            {
                node = make_shared<op::v0::LayerNorm>(
                    args[0], args[1], args[2], keep_stats, begin_norm_axis, epsilon);
            }
            else
            {
                node =
                    make_shared<op::v0::LayerNorm>(args[0], keep_stats, begin_norm_axis, epsilon);
            }
            break;
        }
        case OP_TYPEID::LayerNormBackprop_v0:
        {
            auto use_stats = node_js.at("use_stats").get<bool>();
            auto use_affine = node_js.at("use_affine").get<bool>();
            auto epsilon = node_js.at("epsilon").get<double>();
            auto begin_norm_axis = node_js.at("begin_norm_axis").get<int64_t>();
            if (use_stats && use_affine)
            {
                node = make_shared<op::v0::LayerNormBackprop>(
                    args[0], args[1], args[2], args[3], args[4], begin_norm_axis, epsilon);
            }
            else if (use_stats)
            {
                node = make_shared<op::v0::LayerNormBackprop>(
                    args[0], args[1], args[2], args[3], begin_norm_axis, epsilon);
            }
            else if (use_affine)
            {
                node = make_shared<op::v0::LayerNormBackprop>(
                    args[0], args[1], args[2], begin_norm_axis, epsilon);
            }
            else
            {
                node = make_shared<op::v0::LayerNormBackprop>(
                    args[0], args[1], begin_norm_axis, epsilon);
            }
            break;
        }
        case OP_TYPEID::Less_v0:
        {
            node = make_shared<op::v1::Less>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Less_v1:
        {
            node = make_shared<op::v1::Less>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::LessEq_v0:
        {
            node = make_shared<op::v1::LessEqual>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::LessEqual_v1:
        {
            node = make_shared<op::v1::LessEqual>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Log_v0:
        {
            node = make_shared<op::v0::Log>(args[0]);
            break;
        }
        case OP_TYPEID::LRN_v0:
        {
            auto alpha = node_js.at("alpha").get<double>();
            auto beta = node_js.at("beta").get<double>();
            auto bias = node_js.at("bias").get<double>();
            auto nsize = node_js.at("nsize").get<size_t>();
            node = make_shared<op::v0::LRN>(args[0], args[1], alpha, beta, bias, nsize);
            break;
        }
        case OP_TYPEID::LSTMCell_v0:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto weights_format = get_or_default<op::LSTMWeightsFormat>(
                node_js, "weights_format", op::LSTMWeightsFormat::IFCO);
            auto clip = node_js.at("clip").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activations_alpha = node_js.at("activations_alpha").get<vector<float>>();
            auto activations_beta = node_js.at("activations_beta").get<vector<float>>();
            auto input_forget = node_js.at("input_forget").get<bool>();
            switch (args.size())
            {
            case 7:
                node = make_shared<op::v0::LSTMCell>(args[0],
                                                     args[1],
                                                     args[2],
                                                     args[3],
                                                     args[4],
                                                     args[5],
                                                     args[6],
                                                     hidden_size,
                                                     weights_format,
                                                     activations,
                                                     activations_alpha,
                                                     activations_beta,
                                                     clip,
                                                     input_forget);
                break;
            case 6:
                node = make_shared<op::v0::LSTMCell>(args[0],
                                                     args[1],
                                                     args[2],
                                                     args[3],
                                                     args[4],
                                                     args[5],
                                                     hidden_size,
                                                     weights_format,
                                                     activations,
                                                     activations_alpha,
                                                     activations_beta,
                                                     clip,
                                                     input_forget);
                break;
            case 5:
                node = make_shared<op::v0::LSTMCell>(args[0],
                                                     args[1],
                                                     args[2],
                                                     args[3],
                                                     args[4],
                                                     hidden_size,
                                                     weights_format,
                                                     activations,
                                                     activations_alpha,
                                                     activations_beta,
                                                     clip,
                                                     input_forget);
                break;
            default: throw runtime_error("LSTMCell constructor not supported in serializer");
            }
            break;
        }
        case OP_TYPEID::LSTMSequence_v0:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto clip = node_js.at("clip_threshold").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activations_alpha = node_js.at("activations_alpha").get<vector<float>>();
            auto activations_beta = node_js.at("activations_beta").get<vector<float>>();
            auto input_forget = node_js.at("input_forget").get<bool>();
            auto direction = node_js.at("direction").get<op::v0::LSTMSequence::direction>();
            auto weights_format = get_or_default<op::LSTMWeightsFormat>(
                node_js, "weights_format", op::LSTMWeightsFormat::IFCO);
            if (args.size() == 8)
            {
                node = make_shared<op::v0::LSTMSequence>(args[0],
                                                         args[1],
                                                         args[2],
                                                         args[3],
                                                         args[4],
                                                         args[5],
                                                         args[6],
                                                         args[7],
                                                         hidden_size,
                                                         direction,
                                                         weights_format,
                                                         activations_alpha,
                                                         activations_beta,
                                                         activations,
                                                         clip,
                                                         input_forget);
            }
            else
            {
                node = make_shared<op::v0::LSTMSequence>(args[0],
                                                         args[1],
                                                         args[2],
                                                         args[3],
                                                         args[4],
                                                         args[5],
                                                         args[6],
                                                         hidden_size,
                                                         direction,
                                                         weights_format,
                                                         activations_alpha,
                                                         activations_beta,
                                                         activations,
                                                         clip,
                                                         input_forget);
            }
            break;
        }
        case OP_TYPEID::MatMul_v0:
        {
            bool transpose_a = node_js.at("transpose_a").get<bool>();
            bool transpose_b = node_js.at("transpose_b").get<bool>();
            node = make_shared<op::v0::MatMul>(args[0], args[1], transpose_a, transpose_b);
            break;
        }
        case OP_TYPEID::Max_v0:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::v0::Max>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::MaxPool_v0:
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
                node = make_shared<op::v0::MaxPool>(args[0], window_shape, window_movement_strides);
            }

            break;
        }
        case OP_TYPEID::MaxPoolBackprop_v0:
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
            break;
        }
        case OP_TYPEID::Maximum_v0:
        {
            node = make_shared<op::v1::Maximum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Maximum_v1:
        {
            node = make_shared<op::v1::Maximum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Min_v0:
        {
            auto reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            node = make_shared<op::v0::Min>(args[0], reduction_axes);
            break;
        }
        case OP_TYPEID::Minimum_v0:
        {
            node = make_shared<op::v1::Minimum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Minimum_v1:
        {
            node = make_shared<op::v1::Minimum>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Multiply_v0:
        {
            node = make_shared<op::v1::Multiply>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::MVN_v0:
        {
            auto normalize_variance = node_js.at("normalize_variance").get<bool>();
            AxisSet reduction_axes = deserialize_axis_set(node_js.at("reduction_axes"));
            auto eps = node_js.at("eps").get<double>();
            if (reduction_axes.size() > 0)
            {
                node = make_shared<op::v0::MVN>(args[0], reduction_axes, normalize_variance, eps);
            }
            else
            {
                node = make_shared<op::v0::MVN>(args[0], true, normalize_variance, eps);
            }
            break;
        }
        case OP_TYPEID::Negative_v0:
        {
            node = make_shared<op::v0::Negative>(args[0]);
            break;
        }
        case OP_TYPEID::NonZero_v3:
        {
            auto target_type = read_element_type(node_js.at("index_element_type"));
            node = make_shared<op::v3::NonZero>(args[0], target_type);
            break;
        }
        case OP_TYPEID::NormalizeL2_v0:
        {
            float eps = node_js.at("eps").get<float>();
            auto eps_mode = node_js.at("eps_mode").get<op::EpsMode>();
            node = make_shared<op::v0::NormalizeL2>(args[0], args[1], eps, eps_mode);
            break;
        }
        case OP_TYPEID::NotEqual_v0:
        {
            node = make_shared<op::v1::NotEqual>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::NotEqual_v1:
        {
            node = make_shared<op::v1::NotEqual>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Not_v0:
        {
            node = make_shared<op::v1::LogicalNot>(args[0]);
            break;
        }
        case OP_TYPEID::OneHot_v0:
        {
            if (op_version == 0)
            {
                auto shape = node_js.at("shape").get<vector<size_t>>();
                auto one_hot_axis = node_js.at("one_hot_axis").get<size_t>();
                node =
                    make_shared<op::v0::OneHot>(args[0], read_partial_shape(shape), one_hot_axis);
            }
            if (op_version == 1)
            {
                auto axis = node_js.at("axis").get<int64_t>();
                node = make_shared<op::v1::OneHot>(args[0], args[1], args[2], args[3], axis);
            }
            break;
        }
        case OP_TYPEID::Or_v0:
        {
            node = make_shared<op::v1::LogicalOr>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Pad_v0:
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

            node =
                make_shared<op::v0::Pad>(args[0], args[1], padding_below, padding_above, pad_mode);

            break;
        }
        case OP_TYPEID::Parameter_v0:
        {
            auto type_node_js =
                has_key(node_js, "element_type") ? node_js : node_js.at("value_type");
            auto element_type = read_element_type(type_node_js.at("element_type"));
            auto shape = type_node_js.at("shape");
            auto cacheable = get_or_default<bool>(node_js, "cacheable", false);
            node =
                make_shared<op::v0::Parameter>(element_type, read_partial_shape(shape), cacheable);
            break;
        }
        case OP_TYPEID::PartialSlice_v0:
        {
            auto axes = node_js.at("axes").get<vector<size_t>>();
            auto lower_bounds = node_js.at("lower_bounds").get<vector<int64_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<int64_t>>();
            auto decrease_axes = node_js.at("decrease_axes").get<vector<size_t>>();
            node = make_shared<op::v0::PartialSlice>(
                args[0], axes, lower_bounds, upper_bounds, decrease_axes);
            break;
        }
        case OP_TYPEID::PartialSliceBackprop_v0:
        {
            auto axes = node_js.at("axes").get<vector<size_t>>();
            auto lower_bounds = node_js.at("lower_bounds").get<vector<int64_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<int64_t>>();
            node = make_shared<op::v0::PartialSliceBackprop>(
                args[0], args[1], axes, lower_bounds, upper_bounds);
            break;
        }
        case OP_TYPEID::Passthrough_v0:
        {
            std::vector<json> outputs_js = node_js.at("output_shapes");
            std::vector<std::tuple<element::Type, PartialShape>> outputs;
            for (auto output_js : outputs_js)
            {
                outputs.emplace_back(read_element_type(output_js.at("element_type")),
                                     read_partial_shape(output_js.at("shape")));
            }
            node = make_shared<op::v0::Passthrough>(node_js.at("logical_type"),
                                                    node_js.at("language"),
                                                    node_js.at("function"),
                                                    static_cast<OutputVector>(args),
                                                    std::move(outputs));
            break;
        }
        case OP_TYPEID::Power_v0:
        {
            node = make_shared<op::v1::Power>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Power_v1:
        {
            node = make_shared<op::v1::Power>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::PRelu_v0:
        {
            node = make_shared<op::v0::PRelu>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Product_v0:
        {
            set<size_t> reduction_axes =
                get_or_default<set<size_t>>(node_js, "reduction_axes", set<size_t>());
            if (reduction_axes.empty())
            {
                node = make_shared<op::v0::Product>(args[0], args[1]);
            }
            else
            {
                node = make_shared<op::v0::Product>(args[0], reduction_axes);
            }
            break;
        }
        case OP_TYPEID::PSROIPooling_v0: { break;
        }
        case OP_TYPEID::PriorBox_v0: { break;
        }
        case OP_TYPEID::PriorBoxClustered_v0: { break;
        }
        case OP_TYPEID::Proposal_v0: { break;
        }

        case OP_TYPEID::Quantize_v0:
        {
            auto type = read_element_type(node_js.at("type"));
            auto axes = deserialize_axis_set(node_js.at("axes"));
            auto round_mode = node_js.at("round_mode").get<op::v0::Quantize::RoundMode>();
            node = make_shared<op::v0::Quantize>(args[0], args[1], args[2], type, axes, round_mode);
            break;
        }
        case OP_TYPEID::QuantizedConvolutionBias_v0: { break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasAdd_v0: { break;
        }
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd_v0: { break;
        }
        case OP_TYPEID::QuantizedConvolutionRelu_v0: { break;
        }
        case OP_TYPEID::QuantizedConvolution_v0:
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
            node = make_shared<op::v0::QuantizedConvolution>(
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
        case OP_TYPEID::QuantizedDotBias_v0: { break;
        }
        case OP_TYPEID::QuantizedDot_v0:
        {
            size_t reduction_axes_count = node_js["reduction_axes_count"].get<size_t>();
            auto output_type = read_element_type(node_js.at("output_type"));
            auto input0_axes = node_js.at("input0_axes").get<set<size_t>>();
            auto input1_axes = node_js.at("input1_axes").get<set<size_t>>();
            auto output_axes = node_js.at("output_axes").get<set<size_t>>();

            node = make_shared<op::v0::QuantizedDot>(args[0],
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
        case OP_TYPEID::Recv_v0:
        {
            auto src_id = node_js.at("source_id").get<size_t>();
            node = make_shared<op::v0::Recv>(args[0], src_id);
            break;
        }
        case OP_TYPEID::RandomUniform_v0:
        {
            auto fixed_seed = node_js.at("fixed_seed").get<uint64_t>();
            node =
                make_shared<op::v0::RandomUniform>(args[0], args[1], args[2], args[3], fixed_seed);
            break;
        }
        case OP_TYPEID::Range_v0:
        {
            node = make_shared<op::v0::Range>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Relu_v0:
        {
            node = make_shared<op::v0::Relu>(args[0]);
            break;
        }
        case OP_TYPEID::ReluBackprop_v0:
        {
            node = make_shared<op::v0::ReluBackprop>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::ReplaceSlice_v0:
        {
            auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
            auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::v0::ReplaceSlice>(
                args[0], args[1], lower_bounds, upper_bounds, strides);
            break;
        }
        case OP_TYPEID::Reshape_v0:
        {
            auto input_order = node_js.at("input_order").get<vector<size_t>>();
            auto output_shape = node_js.at("output_shape").get<vector<size_t>>();
            node = make_shared<op::v0::Reshape>(args[0], input_order, output_shape);
            break;
        }
        case OP_TYPEID::Result_v0:
        {
            auto needs_default_layout =
                get_or_default<bool>(node_js, "needs_default_layout", false);
            node = make_shared<op::v0::Result>(args[0], needs_default_layout);
            break;
        }
        case OP_TYPEID::Reverse_v0:
        {
            const auto reversed_axes = deserialize_axis_set(node_js.at("reversed_axes"));
            node = make_shared<op::v0::Reverse>(args[0], reversed_axes);
            break;
        }
        case OP_TYPEID::ReverseSequence_v0:
        {
            auto batch_axis = node_js.at("batch_axis").get<int64_t>();
            auto sequence_axis = node_js.at("sequence_axis").get<int64_t>();
            node =
                make_shared<op::v0::ReverseSequence>(args[0], args[1], batch_axis, sequence_axis);
            break;
        }
        case OP_TYPEID::RNNCell_v0:
        {
            auto hidden_size = node_js.at("hidden_size").get<size_t>();
            auto clip = node_js.at("clip").get<float>();
            auto activations = node_js.at("activations").get<vector<string>>();
            auto activation_alpha = node_js.at("activations_alpha").get<vector<float>>();
            auto activation_beta = node_js.at("activations_beta").get<vector<float>>();
            switch (args.size())
            {
            case 4:
                node = make_shared<op::v0::RNNCell>(args[0],
                                                    args[1],
                                                    args[2],
                                                    args[3],
                                                    hidden_size,
                                                    activations,
                                                    activation_alpha,
                                                    activation_beta,
                                                    clip);
                break;
            case 5:
                node = make_shared<op::v0::RNNCell>(args[0],
                                                    args[1],
                                                    args[2],
                                                    args[3],
                                                    args[4],
                                                    hidden_size,
                                                    activations,
                                                    activation_alpha,
                                                    activation_beta,
                                                    clip);
                break;
            default: throw runtime_error("GRUCell constructor not supported in serializer");
            }
            break;
        }
        case OP_TYPEID::ROIPooling_v0: { break;
        }
        case OP_TYPEID::RegionYolo_v0: { break;
        }
        case OP_TYPEID::ReorgYolo_v0:
        {
            break;
            const auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::v0::ReorgYolo>(args[0], strides);
            break;
        }
        case OP_TYPEID::Round_v0:
        {
            node = make_shared<op::v0::Round>(args[0]);
            break;
        }
        case OP_TYPEID::ScalarConstantLike_v0:
        {
            double value = node_js.at("value").get<double>();
            node = make_shared<op::v0::ScalarConstantLike>(args[0], value);
            break;
        }
        case OP_TYPEID::ScaleShift_v0:
        {
            node = make_shared<op::v0::ScaleShift>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterAdd_v0:
        {
            node = make_shared<op::v0::ScatterAdd>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterND_v0:
        {
            node = make_shared<op::v0::ScatterND>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::ScatterNDAdd_v0:
        {
            node = make_shared<op::v0::ScatterNDAdd>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Select_v0:
        {
            node = make_shared<op::v0::Select>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Stack_v0:
        {
            auto axis = node_js.at("axis").get<size_t>();
            node = make_shared<op::v0::Stack>(static_cast<OutputVector>(args), axis);
            break;
        }
        case OP_TYPEID::Selu_v0:
        {
            node = make_shared<op::v0::Selu>(args[0], args[1], args[2]);
            break;
        }
        case OP_TYPEID::Send_v0:
        {
            auto dest_id = node_js.at("dest_id").get<size_t>();
            node = make_shared<op::v0::Send>(args[0], dest_id);
            break;
        }
        case OP_TYPEID::ShapeOf_v0:
        {
            node = make_shared<op::v0::ShapeOf>(args[0]);
            break;
        }
        case OP_TYPEID::ShuffleChannels_v0:
        {
            const auto axis = node_js.at("axis").get<size_t>();
            const auto group = node_js.at("group").get<size_t>();
            node = make_shared<op::v0::ShuffleChannels>(args[0], axis, group);
            break;
        }
        case OP_TYPEID::Sigmoid_v0:
        {
            node = make_shared<op::v0::Sigmoid>(args[0]);
            break;
        }
        case OP_TYPEID::SigmoidBackprop_v0:
        {
            node = make_shared<op::v0::SigmoidBackprop>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Sign_v0:
        {
            node = make_shared<op::v0::Sign>(args[0]);
            break;
        }
        case OP_TYPEID::Sin_v0:
        {
            node = make_shared<op::v0::Sin>(args[0]);
            break;
        }
        case OP_TYPEID::Sinh_v0:
        {
            node = make_shared<op::v0::Sinh>(args[0]);
            break;
        }
        case OP_TYPEID::Slice_v0:
        {
            auto lower_bounds = node_js.at("lower_bounds").get<vector<size_t>>();
            auto upper_bounds = node_js.at("upper_bounds").get<vector<size_t>>();
            auto strides = node_js.at("strides").get<vector<size_t>>();
            node = make_shared<op::v0::Slice>(args[0], lower_bounds, upper_bounds, strides);
            break;
        }
        case OP_TYPEID::Softmax_v0:
        {
            if (has_key(node_js, "softmax_axes"))
            {
                auto softmax_axes = deserialize_axis_set(node_js.at("softmax_axes"));
                node = make_shared<op::v0::Softmax>(args[0], softmax_axes);
            }
            else
            {
                node = make_shared<op::v0::Softmax>(args[0], args[1]);
            }

            break;
        }
        case OP_TYPEID::SoftmaxCrossEntropy_v0:
        {
            auto soft_label = node_js.at("soft_label");
            auto ignore_index = node_js.at("ignore_index");
            node = make_shared<op::v0::SoftmaxCrossEntropy>(
                args[0], args[1], soft_label, ignore_index);
            break;
        }
        case OP_TYPEID::SoftmaxCrossEntropyBackprop_v0:
        {
            auto soft_label = node_js.at("soft_label");
            auto ignore_index = node_js.at("ignore_index");
            node = make_shared<op::v0::SoftmaxCrossEntropyBackprop>(
                args[0], args[1], args[2], soft_label, ignore_index);
            break;
        }
        case OP_TYPEID::SpaceToDepth_v0:
        {
            auto block_size = node_js.at("block_size").get<size_t>();
            auto mode = node_js.at("mode").get<op::v0::SpaceToDepth::SpaceToDepthMode>();
            node = make_shared<op::v0::SpaceToDepth>(args[0], mode, block_size);
            break;
        }
        case OP_TYPEID::Split_v0:
        {
            const auto splits = node_js.at("splits").get<vector<size_t>>();
            node = make_shared<op::v0::Split>(args[0], args[1], splits);
            break;
        }
        case OP_TYPEID::Sqrt_v0:
        {
            node = make_shared<op::v0::Sqrt>(args[0]);
            break;
        }
        case OP_TYPEID::SquaredDifference_v0:
        {
            node = make_shared<op::v0::SquaredDifference>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Squeeze_v0:
        {
            node = make_shared<op::v0::Squeeze>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Subtract_v0:
        {
            node = make_shared<op::v1::Subtract>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        case OP_TYPEID::Sum_v0:
        {
            set<size_t> reduction_axes =
                get_or_default<set<size_t>>(node_js, "reduction_axes", set<size_t>());
            if (reduction_axes.empty())
            {
                node = make_shared<op::v0::Sum>(args[0], args[1]);
            }
            else
            {
                node = make_shared<op::v0::Sum>(args[0], reduction_axes);
            }
            break;
        }
        case OP_TYPEID::Tan_v0:
        {
            node = make_shared<op::v0::Tan>(args[0]);
            break;
        }
        case OP_TYPEID::Tanh_v0:
        {
            node = make_shared<op::v0::Tanh>(args[0]);
            break;
        }
        case OP_TYPEID::TensorIterator_v0:
        {
            auto ti = make_shared<op::v0::TensorIterator>(args);
            json jbody = node_js["body"];
            // Serializer assumes inputs are available before users sp we
            // need to make sure the body nodes are all deserialized before
            // referencing them.
            json jbody_nodes = jbody["nodes"];
            NodeVector body_nodes;
            for (json jnode : jbody_nodes)
            {
                body_nodes.push_back(deserialize_node(jnode));
            }
            json jparams = jbody["parameters"];
            ParameterVector parameters;
            for (json jparam : jparams)
            {
                parameters.push_back(as_type_ptr<op::v0::Parameter>(deserialize_node(jparam)));
            }
            json jresults = jbody["results"];
            ResultVector results;
            for (json jresult : jresults)
            {
                results.push_back(as_type_ptr<op::v0::Result>(deserialize_node(jresult)));
            }
            ti->set_body(make_shared<op::v0::TensorIterator::BodyLambda>(results, parameters));
            json jins = node_js["input_descriptions"];
            for (json jin : jins)
            {
                ti->get_input_descriptions().push_back(
                    deserialize_tensor_iterator_input_description(jin));
            }
            json jouts = node_js["output_descriptions"];
            for (json jout : jouts)
            {
                ti->get_output_descriptions().push_back(
                    deserialize_tensor_iterator_output_description(jout));
            }
            ti->set_output_size(ti->get_output_descriptions().size());

            node = ti;
            break;
        }

        case OP_TYPEID::Tile_v0:
        {
            node = make_shared<op::v0::Tile>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::TopK_v0:
        {
            auto compute_max = node_js.at("compute_max").get<bool>();
            auto target_type = read_element_type(node_js.at("index_element_type"));
            op::TopKSortType sort =
                get_or_default<op::TopKSortType>(node_js, "sort", op::TopKSortType::value);
            if (has_key(node_js, "top_k_axis"))
            {
                auto top_k_axis = node_js.at("top_k_axis").get<size_t>();
                if (has_key(node_js, "k"))
                {
                    auto k = node_js.at("k").get<size_t>();
                    node = make_shared<op::v0::TopK>(
                        args[0], top_k_axis, target_type, k, compute_max, sort);
                }
                else
                {
                    node = make_shared<op::v0::TopK>(
                        args[0], args[1], top_k_axis, target_type, compute_max, sort);
                }
            }
            else
            {
                node = make_shared<op::v0::TopK>(
                    args[0], args[1], args[2], target_type, compute_max, sort);
            }
            break;
        }
        case OP_TYPEID::StopGradient_v0:
        {
            node = make_shared<op::v0::StopGradient>(args[0]);
            break;
        }
        case OP_TYPEID::Unsqueeze_v0:
        {
            node = make_shared<op::v0::Unsqueeze>(args[0], args[1]);
            break;
        }
        case OP_TYPEID::Xor_v0:
        {
            node = make_shared<op::v1::LogicalXor>(
                args[0], args[1], read_auto_broadcast(node_js, "auto_broadcast"));
            break;
        }
        default:
        {
            stringstream ss;
            ss << "unsupported op " << type_info.name << ":" << type_info.version;
            throw runtime_error(ss.str());
        }
        }
#if defined(__GNUC__) && !(__GNUC__ == 4 && __GNUC_MINOR__ == 8)
#pragma GCC diagnostic pop
#endif

        if (node)
        {
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
                if (has_key(node_js, "provenance_tags"))
                {
                    const std::vector<json> prov_js = node_js.at("provenance_tags");
                    for (auto prov_tag : prov_js)
                    {
                        node->add_provenance_tag(prov_tag);
                    }
                }
            }
            m_node_map[node_name] = node;
        }
    }
    catch (exception& err)
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

json JSONSerializer::serialize_output(const Output<Node>& output)
{
    json result;
    auto index = output.get_index();
    json json_node_reference = output.get_node()->get_name();
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
    const NodeTypeInfo& type_info = n.get_type_info();
    json node;
    node["name"] = n.get_name();
    auto op_version = type_info.version;
    node["op_version"] = op_version;

    if (n.get_name() != n.get_friendly_name())
    {
        node["friendly_name"] = n.get_friendly_name();
    }
    node["op"] = type_info.name;
    // TODO Multiple outputs
    json inputs = json::array();
    json control_deps = json::array();
    json outputs = json::array();

    string op_full_name = string(type_info.name) + "_v" + to_string(op_version);

    for (auto& input : n.inputs())
    {
        inputs.push_back(serialize_output(input.get_source_output()));
    }
    for (auto cdep : n.get_control_dependencies())
    {
        control_deps.push_back(cdep->get_name());
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

    if (ngraph::get_provenance_enabled())
    {
        json provenance_tags = json::array();
        for (auto prov_tag : n.get_provenance_tags())
        {
            provenance_tags.push_back(prov_tag);
        }
        node["provenance_tags"] = provenance_tags;
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
// #pragma GCC diagnostic error "-Wswitch-enum"
// #pragma GCC diagnostic error "-Wimplicit-fallthrough"
#endif
    switch (get_typeid(op_full_name))
    {
    case OP_TYPEID::Abs_v0: { break;
    }
    case OP_TYPEID::Acos_v0: { break;
    }
    case OP_TYPEID::ArgMin_v0:
    {
        auto tmp = static_cast<const op::v0::ArgMin*>(&n);
        node["axis"] = tmp->get_reduction_axis();
        node["index_element_type"] = write_element_type(tmp->get_output_element_type(0));
        break;
    }
    case OP_TYPEID::ArgMax_v0:
    {
        auto tmp = static_cast<const op::v0::ArgMax*>(&n);
        node["axis"] = tmp->get_reduction_axis();
        node["index_element_type"] = write_element_type(tmp->get_output_element_type(0));
        break;
    }
    case OP_TYPEID::All_v0:
    {
        auto tmp = static_cast<const op::v0::All*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::AllReduce_v0: { break;
    }
    case OP_TYPEID::Any_v0:
    {
        auto tmp = static_cast<const op::v0::Any*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::Asin_v0: { break;
    }
    case OP_TYPEID::Atan_v0: { break;
    }
    case OP_TYPEID::Atan2_v0:
    {
        auto tmp = dynamic_cast<const op::v0::Atan2*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["autob"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::AvgPool_v0:
    {
        auto tmp = static_cast<const op::v0::AvgPool*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["include_padding_in_avg_computation"] = tmp->get_include_padding_in_avg_computation();
        node["pad_type"] = tmp->get_pad_type();
        if (tmp->get_ceil_mode())
        {
            node["ceil_mode"] = tmp->get_ceil_mode();
        }
        break;
    }
    case OP_TYPEID::AvgPoolBackprop_v0:
    {
        auto tmp = static_cast<const op::v0::AvgPoolBackprop*>(&n);
        node["forward_arg_shape"] = tmp->get_forward_arg_shape();
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["include_padding_in_avg_computation"] = tmp->get_include_padding_in_avg_computation();
        break;
    }
    case OP_TYPEID::BatchMatMul_v0: { break;
    }
    case OP_TYPEID::BatchMatMulTranspose_v0:
    {
        auto tmp = static_cast<const op::v0::BatchMatMulTranspose*>(&n);
        node["transpose_0"] = tmp->get_transpose_arg0();
        node["transpose_1"] = tmp->get_transpose_arg1();
        break;
    }
    case OP_TYPEID::BatchNormTraining_v0:
    {
        auto tmp = static_cast<const op::v0::BatchNormTraining*>(&n);
        node["eps"] = tmp->get_eps_value();
        break;
    }
    case OP_TYPEID::BatchNormInference_v0:
    {
        auto tmp = static_cast<const op::v0::BatchNormInference*>(&n);
        node["eps"] = tmp->get_eps_value();
        break;
    }
    case OP_TYPEID::BatchNormTrainingBackprop_v0:
    {
        auto tmp = static_cast<const op::v0::BatchNormTrainingBackprop*>(&n);
        node["eps"] = tmp->get_eps_value();
        break;
    }
    case OP_TYPEID::Broadcast_v0:
    {
        auto tmp = dynamic_cast<const op::v0::Broadcast*>(&n);
        node["axes"] = serialize_axis_set(tmp->get_broadcast_axes());
        node["shape"] = tmp->get_broadcast_shape();
        break;
    }
    case OP_TYPEID::BroadcastDistributed_v0: { break;
    }
    case OP_TYPEID::BroadcastLike_v0:
    {
        auto tmp = static_cast<const op::v0::BroadcastLike*>(&n);
        node["initial_axes"] = serialize_axis_set(tmp->get_initial_broadcast_axes());
        break;
    }
    case OP_TYPEID::Ceiling_v0: { break;
    }
    case OP_TYPEID::Clamp_v0:
    {
        auto tmp = static_cast<const op::v0::Clamp*>(&n);
        node["min"] = to_cpp_string<double>(tmp->get_min<double>());
        node["max"] = to_cpp_string<double>(tmp->get_max<double>());
        break;
    }
    case OP_TYPEID::Concat_v0:
    {
        auto tmp = static_cast<const op::v0::Concat*>(&n);
        node["axis"] = tmp->get_concatenation_axis();
        break;
    }
    case OP_TYPEID::Constant_v0:
    {
        auto tmp = static_cast<const op::v0::Constant*>(&n);
        if (tmp->get_all_data_elements_bitwise_identical() &&
            shape_size(tmp->get_output_shape(0)) > 0)
        {
            vector<string> vs;
            vs.push_back(tmp->convert_value_to_string(0));
            node["value"] = vs;
        }
        else
        {
            node["value"] = tmp->get_value_strings();
        }
        node["shape"] = tmp->get_output_shape(0);
        node["element_type"] = write_element_type(tmp->get_output_element_type(0));
        break;
    }
    case OP_TYPEID::Convert_v0:
    {
        auto tmp = static_cast<const op::v0::Convert*>(&n);
        node["target_type"] = write_element_type(tmp->get_convert_element_type());
        break;
    }
    case OP_TYPEID::Convolution_v0:
    {
        auto tmp = static_cast<const op::v0::Convolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        node["pad_type"] = tmp->get_pad_type();
        break;
    }
    case OP_TYPEID::ConvolutionBackpropData_v0:
    {
        auto tmp = static_cast<const op::v0::ConvolutionBackpropData*>(&n);
        node["data_batch_shape"] = tmp->get_data_batch_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
        break;
    }
    case OP_TYPEID::ConvolutionBackpropFilters_v0:
    {
        auto tmp = static_cast<const op::v0::ConvolutionBackpropFilters*>(&n);
        node["filters_shape"] = tmp->get_filters_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
        break;
    }
    case OP_TYPEID::ConvolutionBias_v0:
    {
        auto tmp = static_cast<const op::v0::ConvolutionBias*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        break;
    }
    case OP_TYPEID::ConvolutionBiasAdd_v0:
    {
        auto tmp = static_cast<const op::v0::ConvolutionBiasAdd*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        break;
    }
    case OP_TYPEID::ConvolutionBiasBackpropFiltersBias_v0:
    {
        auto tmp = static_cast<const op::v0::ConvolutionBiasBackpropFiltersBias*>(&n);
        node["filters_shape"] = tmp->get_filters_shape();
        node["bias_shape"] = tmp->get_bias_shape();
        node["window_movement_strides_forward"] = tmp->get_window_movement_strides_forward();
        node["window_dilation_strides_forward"] = tmp->get_window_dilation_strides_forward();
        node["padding_below_forward"] = tmp->get_padding_below_forward();
        node["padding_above_forward"] = tmp->get_padding_above_forward();
        node["data_dilation_strides_forward"] = tmp->get_data_dilation_strides_forward();
        break;
    }
    case OP_TYPEID::Cos_v0: { break;
    }
    case OP_TYPEID::Cosh_v0: { break;
    }
    case OP_TYPEID::CumSum_v0:
    {
        auto tmp = static_cast<const op::v0::CumSum*>(&n);
        node["exclusive"] = tmp->is_exclusive();
        node["reverse"] = tmp->is_reverse();
        break;
    }
    case OP_TYPEID::CrossEntropy_v0:
    {
        auto tmp = static_cast<const op::v0::CrossEntropy*>(&n);
        node["soft_label"] = tmp->get_soft_label();
        node["ignore_index"] = tmp->get_ignore_index();
        break;
    }
    case OP_TYPEID::CrossEntropyBackprop_v0:
    {
        auto tmp = static_cast<const op::v0::CrossEntropyBackprop*>(&n);
        node["soft_label"] = tmp->get_soft_label();
        node["ignore_index"] = tmp->get_ignore_index();
        break;
    }
    case OP_TYPEID::CropAndResize_v0:
    {
        auto tmp = static_cast<const op::v0::CropAndResize*>(&n);
        node["resize_method"] = as_string(tmp->get_resize_method());
        node["extrapolation_value"] = tmp->get_extrapolation_value();
        break;
    }
    case OP_TYPEID::CTCGreedyDecoder_v0: { break;
    }
    case OP_TYPEID::DetectionOutput_v0: { break;
    }
    case OP_TYPEID::PSROIPooling_v0: { break;
    }
    case OP_TYPEID::PriorBox_v0: { break;
    }
    case OP_TYPEID::PriorBoxClustered_v0: { break;
    }
    case OP_TYPEID::Proposal_v0: { break;
    }
    case OP_TYPEID::ROIPooling_v0: { break;
    }
    case OP_TYPEID::RegionYolo_v0: { break;
    }
    case OP_TYPEID::ReorgYolo_v0:
    {
        auto tmp = static_cast<const op::v0::ReorgYolo*>(&n);
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::Round_v0: { break;
    }
    case OP_TYPEID::Dequantize_v0:
    {
        auto tmp = static_cast<const op::v0::Dequantize*>(&n);
        node["type"] = write_element_type(tmp->get_output_element_type(0));
        node["axes"] = serialize_axis_set(tmp->get_axes());
        break;
    }
    case OP_TYPEID::DepthToSpace_v0:
    {
        auto tmp = static_cast<const op::v0::DepthToSpace*>(&n);
        node["type"] = write_element_type(tmp->get_output_element_type(0));
        node["mode"] = tmp->get_mode();
        node["block_size"] = tmp->get_block_size();
        break;
    }
    case OP_TYPEID::Dot_v0:
    {
        auto tmp = static_cast<const op::v0::Dot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
        break;
    }
    case OP_TYPEID::DynBroadcast_v0: { break;
    }
    case OP_TYPEID::DynPad_v0: { break;
    }
    case OP_TYPEID::DynReplaceSlice_v0:
    {
        auto tmp = static_cast<const op::v0::DynReplaceSlice*>(&n);
        node["lower_bounds_mask"] = tmp->get_lower_bounds_mask();
        node["upper_bounds_mask"] = tmp->get_upper_bounds_mask();
        node["new_axis"] = tmp->get_new_axis();
        node["shrink_axis"] = tmp->get_shrink_axis();
        node["ellipsis_mask"] = tmp->get_ellipsis_mask();
        break;
    }
    case OP_TYPEID::DynSlice_v0:
    {
        auto tmp = static_cast<const op::v0::DynSlice*>(&n);
        node["lower_bounds_mask"] = tmp->get_lower_bounds_mask();
        node["upper_bounds_mask"] = tmp->get_upper_bounds_mask();
        node["new_axis"] = tmp->get_new_axis();
        node["shrink_axis"] = tmp->get_shrink_axis();
        node["ellipsis_mask"] = tmp->get_ellipsis_mask();
        break;
    }
    case OP_TYPEID::Elu_v0:
    {
        auto tmp = static_cast<const op::v0::Elu*>(&n);
        node["alpha"] = tmp->get_alpha();
        break;
    }
    case OP_TYPEID::EmbeddingLookup_v0: { break;
    }
    case OP_TYPEID::Equal_v1:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v1::Equal*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Erf_v0: { break;
    }
    case OP_TYPEID::Exp_v0: { break;
    }
    case OP_TYPEID::FakeQuantize_v0:
    {
        auto tmp = static_cast<const op::v0::FakeQuantize*>(&n);
        node["levels"] = tmp->get_levels();
        break;
    }
    case OP_TYPEID::Floor_v0: { break;
    }
    case OP_TYPEID::Gather_v0:
    {
        auto tmp = static_cast<const op::v0::Gather*>(&n);
        node["axis"] = tmp->get_axis();
        break;
    }
    case OP_TYPEID::GatherND_v0: { break;
    }
    case OP_TYPEID::GetOutputElement_v0: { break;
    }
    case OP_TYPEID::Gelu_v0: { break;
    }
    case OP_TYPEID::GeluBackpropFactor_v0: { break;
    }
    case OP_TYPEID::Gemm_v0:
    {
        auto tmp = static_cast<const op::v0::Gemm*>(&n);
        node["alpha"] = tmp->get_alpha();
        node["beta"] = tmp->get_beta();
        node["transA"] = tmp->get_transA();
        node["transB"] = tmp->get_transB();
        break;
    }
    case OP_TYPEID::GenerateMask_v0:
    {
        auto tmp = static_cast<const op::v0::GenerateMask*>(&n);
        node["type"] = write_element_type(tmp->get_element_type());
        node["use_seed"] = tmp->get_use_seed();
        node["seed"] = tmp->get_seed();
        node["probability"] = tmp->get_probability();
        node["output_shape"] = tmp->get_mask_shape();
        break;
    }
    case OP_TYPEID::Greater_v1:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v1::Greater*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::GRN_v0:
    {
        auto tmp = static_cast<const op::v0::GRN*>(&n);
        node["bias"] = tmp->get_bias();
        break;
    }
    case OP_TYPEID::GroupConvolution_v0:
    {
        auto tmp = static_cast<const op::v0::GroupConvolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        if (!tmp->has_groups_in_filters())
        {
            node["groups"] = tmp->get_groups();
        }
        node["pad_type"] = tmp->get_pad_type();
        break;
    }
    case OP_TYPEID::GroupConvolutionBackpropData_v0:
    {
        auto tmp = static_cast<const op::v0::GroupConvolutionBackpropData*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["groups"] = tmp->get_groups();
        break;
    }
    case OP_TYPEID::GroupConvolutionBackpropFilters_v0:
    {
        auto tmp = static_cast<const op::v0::GroupConvolutionBackpropFilters*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["groups"] = tmp->get_groups();
        break;
    }
    case OP_TYPEID::HardSigmoid_v0: { break;
    }
    case OP_TYPEID::LayerNorm_v0:
    {
        auto tmp = static_cast<const op::v0::LayerNorm*>(&n);
        node["keep_stats"] = tmp->get_keep_stats();
        node["use_affine"] = tmp->get_use_affine();
        node["epsilon"] = tmp->get_epsilon();
        node["begin_norm_axis"] = tmp->get_begin_norm_axis();
        break;
    }
    case OP_TYPEID::LayerNormBackprop_v0:
    {
        auto tmp = static_cast<const op::v0::LayerNormBackprop*>(&n);
        node["use_stats"] = tmp->get_use_stats();
        node["use_affine"] = tmp->get_use_affine();
        node["epsilon"] = tmp->get_epsilon();
        node["begin_norm_axis"] = tmp->get_begin_norm_axis();
        break;
    }
    case OP_TYPEID::Less_v1:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v1::Less*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::LessEqual_v1:
    {
        auto tmp = static_cast<const op::v1::LessEqual*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Log_v0: { break;
    }
    case OP_TYPEID::LRN_v0:
    {
        auto tmp = static_cast<const op::v0::LRN*>(&n);
        node["alpha"] = tmp->get_alpha();
        node["beta"] = tmp->get_beta();
        node["bias"] = tmp->get_bias();
        node["nsize"] = tmp->get_nsize();
        break;
    }
    case OP_TYPEID::LSTMCell_v0:
    {
        auto tmp = static_cast<const op::v0::LSTMCell*>(&n);
        node["hidden_size"] = tmp->get_hidden_size();
        node["weights_format"] = tmp->get_weights_format();
        node["clip"] = tmp->get_clip();
        node["activations"] = tmp->get_activations();
        node["activations_alpha"] = tmp->get_activations_alpha();
        node["activations_beta"] = tmp->get_activations_beta();
        node["input_forget"] = tmp->get_input_forget();
        break;
    }
    case OP_TYPEID::LSTMSequence_v0:
    {
        auto tmp = dynamic_cast<const op::v0::LSTMSequence*>(&n);
        node["direction"] = tmp->get_direction();
        node["hidden_size"] = tmp->get_hidden_size();
        node["weights_format"] = tmp->get_weights_format();
        node["clip_threshold"] = tmp->get_clip_threshold();
        node["activations"] = tmp->get_activations();
        node["activations_alpha"] = tmp->get_activations_alpha();
        node["activations_beta"] = tmp->get_activations_beta();
        node["input_forget"] = tmp->get_input_forget();
        break;
    }
    case OP_TYPEID::MatMul_v0:
    {
        auto tmp = static_cast<const op::v0::MatMul*>(&n);
        node["transpose_a"] = tmp->get_transpose_a();
        node["transpose_b"] = tmp->get_transpose_b();
        break;
    }
    case OP_TYPEID::Max_v0:
    {
        auto tmp = static_cast<const op::v0::Max*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::MaxPool_v0:
    {
        auto tmp = static_cast<const op::v0::MaxPool*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["pad_type"] = tmp->get_pad_type();
        break;
    }
    case OP_TYPEID::MaxPoolBackprop_v0:
    {
        auto tmp = static_cast<const op::v0::MaxPoolBackprop*>(&n);
        node["window_shape"] = tmp->get_window_shape();
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        break;
    }
    case OP_TYPEID::Maximum_v1:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v1::Maximum*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Min_v0:
    {
        auto tmp = static_cast<const op::v0::Min*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        break;
    }
    case OP_TYPEID::Minimum_v1:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v1::Minimum*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::MVN_v0:
    {
        auto tmp = static_cast<const op::v0::MVN*>(&n);
        node["reduction_axes"] = serialize_axis_set(tmp->get_reduction_axes());
        node["normalize_variance"] = tmp->get_normalize_variance();
        node["eps"] = tmp->get_eps();
        break;
    }
    case OP_TYPEID::Negative_v0: { break;
    }
    case OP_TYPEID::NormalizeL2_v0:
    {
        auto tmp = static_cast<const op::v0::NormalizeL2*>(&n);
        node["eps"] = tmp->get_eps();
        node["eps_mode"] = tmp->get_eps_mode();
        break;
    }
    case OP_TYPEID::NotEqual_v1:
    {
        const op::util::BinaryElementwiseComparison* tmp = nullptr;
        tmp = static_cast<const op::v1::NotEqual*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::OneHot_v0:
    {
        if (op_version == 0)
        {
            auto tmp = static_cast<const op::v0::OneHot*>(&n);
            node["shape"] = write_partial_shape(tmp->get_output_partial_shape(0));
            node["one_hot_axis"] = tmp->get_one_hot_axis();
        }
        if (op_version == 1)
        {
            auto tmp = static_cast<const op::v1::OneHot*>(&n);
            node["axis"] = tmp->get_axis();
        }
        break;
    }
    case OP_TYPEID::Pad_v0:
    {
        auto tmp = static_cast<const op::v0::Pad*>(&n);
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["pad_mode"] = tmp->get_pad_mode();
        break;
    }
    case OP_TYPEID::Parameter_v0:
    {
        auto tmp = static_cast<const op::v0::Parameter*>(&n);
        node["shape"] = write_partial_shape(tmp->get_output_partial_shape(0));
        node["cacheable"] = tmp->get_cacheable();
        node["element_type"] = write_element_type(tmp->get_element_type());
        break;
    }
    case OP_TYPEID::PartialSlice_v0:
    {
        auto tmp = dynamic_cast<const op::v0::PartialSlice*>(&n);
        node["axes"] = tmp->get_axes();
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["decrease_axes"] = tmp->get_decrease_axes();
        break;
    }
    case OP_TYPEID::PartialSliceBackprop_v0:
    {
        auto tmp = dynamic_cast<const op::v0::PartialSliceBackprop*>(&n);
        node["axes"] = tmp->get_axes();
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        break;
    }
    case OP_TYPEID::Passthrough_v0:
    {
        auto tmp = static_cast<const op::v0::Passthrough*>(&n);
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
    case OP_TYPEID::PRelu_v0: { break;
    }
    case OP_TYPEID::Product_v0:
    {
        auto tmp = static_cast<const op::v0::Product*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::Power_v1:
    {
        const op::util::BinaryElementwiseArithmetic* tmp = nullptr;
        tmp = static_cast<const op::v1::Power*>(&n);
        if (tmp != nullptr && tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Quantize_v0:
    {
        auto tmp = static_cast<const op::v0::Quantize*>(&n);
        node["type"] = write_element_type(tmp->get_output_element_type(0));
        node["axes"] = serialize_axis_set(tmp->get_axes());
        node["round_mode"] = tmp->get_round_mode();
        break;
    }
    case OP_TYPEID::QuantizedConvolutionBias_v0: { break;
    }
    case OP_TYPEID::QuantizedConvolutionBiasAdd_v0: { break;
    }
    case OP_TYPEID::QuantizedConvolutionBiasSignedAdd_v0: { break;
    }
    case OP_TYPEID::QuantizedConvolutionRelu_v0: { break;
    }
    case OP_TYPEID::QuantizedConvolution_v0:
    {
        auto tmp = static_cast<const op::v0::QuantizedConvolution*>(&n);
        node["window_movement_strides"] = tmp->get_window_movement_strides();
        node["window_dilation_strides"] = tmp->get_window_dilation_strides();
        node["padding_below"] = tmp->get_padding_below();
        node["padding_above"] = tmp->get_padding_above();
        node["data_dilation_strides"] = tmp->get_data_dilation_strides();
        node["output_type"] = write_element_type(tmp->get_output_element_type(0));
        node["input_axes"] = tmp->get_input_axes();
        node["filter_axes"] = tmp->get_filter_axes();
        node["output_axes"] = tmp->get_output_axes();
        break;
    }
    case OP_TYPEID::QuantizedDotBias_v0: { break;
    }
    case OP_TYPEID::QuantizedDot_v0:
    {
        auto tmp = static_cast<const op::v0::QuantizedDot*>(&n);
        node["reduction_axes_count"] = tmp->get_reduction_axes_count();
        node["output_type"] = write_element_type(tmp->get_output_element_type(0));
        node["input0_axes"] = tmp->get_input0_axes();
        node["input1_axes"] = tmp->get_input1_axes();
        node["output_axes"] = tmp->get_output_axes();
        break;
    }
    case OP_TYPEID::RandomUniform_v0:
    {
        auto tmp = static_cast<const op::v0::RandomUniform*>(&n);
        node["fixed_seed"] = tmp->get_fixed_seed();
        break;
    }
    case OP_TYPEID::Range_v0: { break;
    }
    case OP_TYPEID::Recv_v0:
    {
        auto tmp = static_cast<const op::v0::Recv*>(&n);
        node["source_id"] = tmp->get_src_id();
        break;
    }
    case OP_TYPEID::Relu_v0: { break;
    }
    case OP_TYPEID::ReluBackprop_v0: { break;
    }
    case OP_TYPEID::ReplaceSlice_v0:
    {
        auto tmp = static_cast<const op::v0::ReplaceSlice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::Reshape_v0:
    {
        auto tmp = static_cast<const op::v0::Reshape*>(&n);
        node["input_order"] = tmp->get_input_order();
        node["output_shape"] = tmp->get_output_shape(0);
        break;
    }
    case OP_TYPEID::Result_v0:
    {
        auto tmp = static_cast<const op::v0::Result*>(&n);
        node["needs_default_layout"] = tmp->needs_default_layout();
        break;
    }
    case OP_TYPEID::Reverse_v0:
    {
        const auto tmp = static_cast<const op::v0::Reverse*>(&n);
        node["reversed_axes"] = serialize_axis_set(tmp->get_reversed_axes());
        break;
    }
    case OP_TYPEID::ReverseSequence_v0:
    {
        auto tmp = static_cast<const op::v0::ReverseSequence*>(&n);
        node["batch_axis"] = tmp->get_origin_batch_axis();
        node["sequence_axis"] = tmp->get_origin_sequence_axis();
        break;
    }
    case OP_TYPEID::RNNCell_v0:
    {
        auto tmp = static_cast<const op::v0::RNNCell*>(&n);
        node["hidden_size"] = tmp->get_hidden_size();
        node["clip"] = tmp->get_clip();
        node["activations"] = tmp->get_activations();
        node["activations_alpha"] = tmp->get_activations_alpha();
        node["activations_beta"] = tmp->get_activations_beta();
        break;
    }
    case OP_TYPEID::ScalarConstantLike_v0:
    {
        auto tmp = static_cast<const op::v0::ScalarConstantLike*>(&n);
        auto constant = tmp->as_constant();
        char* p_end;
        node["value"] = strtod(constant->get_value_strings()[0].c_str(), &p_end);
        break;
    }
    case OP_TYPEID::ScaleShift_v0: { break;
    }
    case OP_TYPEID::ScatterAdd_v0: { break;
    }
    case OP_TYPEID::ScatterND_v0: { break;
    }
    case OP_TYPEID::ScatterNDAdd_v0: { break;
    }
    case OP_TYPEID::Select_v0: { break;
    }
    case OP_TYPEID::Selu_v0: { break;
    }
    case OP_TYPEID::Send_v0:
    {
        auto tmp = static_cast<const op::v0::Send*>(&n);
        node["dest_id"] = tmp->get_dest_id();
        break;
    }
    case OP_TYPEID::ShapeOf_v0: { break;
    }
    case OP_TYPEID::ShuffleChannels_v0:
    {
        const auto tmp = static_cast<const op::v0::ShuffleChannels*>(&n);
        node["axis"] = tmp->get_axis();
        node["group"] = tmp->get_group();
        break;
    }
    case OP_TYPEID::Sigmoid_v0: { break;
    }
    case OP_TYPEID::SigmoidBackprop_v0: { break;
    }
    case OP_TYPEID::Sign_v0: { break;
    }
    case OP_TYPEID::Sin_v0: { break;
    }
    case OP_TYPEID::Sinh_v0: { break;
    }
    case OP_TYPEID::Slice_v0:
    {
        auto tmp = static_cast<const op::v0::Slice*>(&n);
        node["lower_bounds"] = tmp->get_lower_bounds();
        node["upper_bounds"] = tmp->get_upper_bounds();
        node["strides"] = tmp->get_strides();
        break;
    }
    case OP_TYPEID::SpaceToDepth_v0:
    {
        auto tmp = static_cast<const op::v0::SpaceToDepth*>(&n);
        node["type"] = write_element_type(tmp->get_output_element_type(0));
        node["mode"] = tmp->get_mode();
        node["block_size"] = tmp->get_block_size();
        break;
    }
    case OP_TYPEID::Split_v0:
    {
        const auto tmp = static_cast<const op::v0::Split*>(&n);
        node["splits"] = tmp->get_splits();
        break;
    }
    case OP_TYPEID::Sqrt_v0: { break;
    }
    case OP_TYPEID::SquaredDifference_v0:
    {
        auto tmp = static_cast<const op::v0::SquaredDifference*>(&n);
        if (tmp->get_autob().m_type != op::AutoBroadcastType::NONE)
        {
            node["auto_broadcast"] = write_auto_broadcast(tmp->get_autob());
        }
        break;
    }
    case OP_TYPEID::Squeeze_v0: { break;
    }
    case OP_TYPEID::StopGradient_v0: { break;
    }
    case OP_TYPEID::Sum_v0:
    {
        auto tmp = static_cast<const op::v0::Sum*>(&n);
        node["reduction_axes"] = tmp->get_reduction_axes();
        break;
    }
    case OP_TYPEID::Stack_v0:
    {
        auto tmp = static_cast<const op::v0::Stack*>(&n);
        node["axis"] = tmp->get_axis();
        break;
    }
    case OP_TYPEID::Softmax_v0: { break;
    }
    case OP_TYPEID::SoftmaxCrossEntropy_v0:
    {
        auto tmp = static_cast<const op::v0::SoftmaxCrossEntropy*>(&n);
        node["soft_label"] = tmp->get_soft_label();
        node["ignore_index"] = tmp->get_ignore_index();
        break;
    }
    case OP_TYPEID::SoftmaxCrossEntropyBackprop_v0:
    {
        auto tmp = static_cast<const op::v0::SoftmaxCrossEntropyBackprop*>(&n);
        node["soft_label"] = tmp->get_soft_label();
        node["ignore_index"] = tmp->get_ignore_index();
        break;
    }
    case OP_TYPEID::Tan_v0: { break;
    }
    case OP_TYPEID::Tanh_v0: { break;
    }
    case OP_TYPEID::TensorIterator_v0:
    {
        auto tmp = static_cast<const op::v0::TensorIterator*>(&n);
        json body = json::object();
        {
            auto& body_results = tmp->get_body()->get_results();
            // Serializer assumes node inputs are already serialized, so
            // we need to capture the body-referenced nodes here.
            json body_nodes = json::array();
            for (auto n : topological_sort(body_results))
            {
                body_nodes.push_back(serialize_node(*n));
            }
            body["nodes"] = body_nodes;
            json params = json::array();
            for (auto param : tmp->get_body()->get_parameters())
            {
                params.push_back(serialize_node(*param));
            }
            body["parameters"] = params;
            json results = json::array();
            for (auto result : body_results)
            {
                results.push_back(serialize_node(*result));
            }
            body["results"] = results;
        }
        node["body"] = body;
        json ins = json::array();
        for (auto in : tmp->get_input_descriptions())
        {
            ins.push_back(serialize_tensor_iterator_input_description(in));
        }
        node["input_descriptions"] = ins;
        json outs = json::array();
        for (auto out : tmp->get_output_descriptions())
        {
            outs.push_back(serialize_tensor_iterator_output_description(out));
        }
        node["output_descriptions"] = outs;
        break;
    }
    case OP_TYPEID::Tile_v0: { break;
    }
    case OP_TYPEID::TopK_v0:
    {
        const auto tmp = static_cast<const op::v0::TopK*>(&n);
        node["index_element_type"] = write_element_type(tmp->get_index_element_type());
        node["compute_max"] = tmp->get_compute_max();
        node["sort"] = tmp->get_sort();
        switch (tmp->inputs().size())
        {
        case 1:
            node["k"] = tmp->get_k();
            node["top_k_axis"] = tmp->get_top_k_axis();
            break;
        case 2: node["top_k_axis"] = tmp->get_top_k_axis(); break;
        case 3: break;
        default: throw runtime_error("TopK constructor not supported in serializer");
        }
        break;
    }
    case OP_TYPEID::Unsqueeze_v0: { break;
    }
    case OP_TYPEID::UnknownOp:
    default:
    {
        auto& factory_registry = FactoryRegistry<Node>::get();
        if (factory_registry.has_factory(type_info))
        {
            node["attribute_visitor"] = true;
            JSONAttributeSerializer visitor(node);
            if (!const_cast<Node&>(n).visit_attributes(visitor))
            {
                NGRAPH_ERR << "Cannot serialize: "
                           << "v" << n.get_type_info().version << "::" << n.get_type_info().name;
            }
            return node;
        }
        else
        {
            NGRAPH_ERR << "Cannot serialize, no factory found: "
                       << "v" << n.get_type_info().version << "::" << n.get_type_info().name;
        }

        break;
    }
    }
#if !(defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ == 8))
#pragma GCC diagnostic pop
#endif
    return node;
}
