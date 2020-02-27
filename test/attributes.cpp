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

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset1.hpp"

using namespace std;
using namespace ngraph;

enum class TuringModel
{
    XL400,
    XL1200
};

namespace ngraph
{
    template <>
    EnumNames<TuringModel>& EnumNames<TuringModel>::get()
    {
        static auto enum_names = EnumNames<TuringModel>(
            "TuringModel", {{"XL400", TuringModel::XL400}, {"XL1200", TuringModel::XL1200}});
        return enum_names;
    }

    template <>
    class AttributeAdapter<TuringModel> : public EnumAttributeAdapterBase<TuringModel>
    {
    public:
        AttributeAdapter(TuringModel& value)
            : EnumAttributeAdapterBase<TuringModel>(value)
        {
        }

        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<TuringModel>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
    };

    constexpr DiscreteTypeInfo AttributeAdapter<TuringModel>::type_info;
}

// Given a Turing machine program and data, return scalar 1 if the program would
// complete, 1 if it would not.
class Oracle : public op::Op
{
public:
    Oracle(const Output<Node>& program,
           const Output<Node>& data,
           TuringModel turing_model,
           uint64_t model_version,
           uint8_t rev,
           const string& serial_number,
           bool enable_turbo,
           const std::vector<uint64_t>& hyper_parameters,
           const std::vector<int64_t>& ultra_parameters)
        : Op({program, data})
        , m_turing_model(turing_model)
        , m_model_version(model_version)
        , m_rev(rev)
        , m_serial_number(serial_number)
        , m_enable_turbo(enable_turbo)
        , m_hyper_parameters(hyper_parameters)
        , m_ultra_parameters(ultra_parameters)
    {
    }

    static constexpr NodeTypeInfo type_info{"Oracle", 0};
    const NodeTypeInfo& get_type_info() const override { return type_info; }
    Oracle() = default;

    TuringModel get_turing_model() const { return m_turing_model; }
    uint64_t get_model_version() const { return m_model_version; }
    const string& get_serial_number() const { return m_serial_number; }
    bool get_enable_turbo() const { return m_enable_turbo; }
    const vector<uint64_t>& get_hyper_parameters() const { return m_hyper_parameters; }
    const vector<int64_t>& get_ultra_parameters() const { return m_ultra_parameters; }
    shared_ptr<Node> copy_with_new_args(const NodeVector& args) const override
    {
        return make_shared<Oracle>(args[0],
                                   args[1],
                                   m_turing_model,
                                   m_model_version,
                                   m_rev,
                                   m_serial_number,
                                   m_enable_turbo,
                                   m_hyper_parameters,
                                   m_ultra_parameters);
    }

    void validate_and_infer_types() override { set_output_type(0, element::i64, {}); }
    bool visit_attributes(AttributeVisitor& visitor) override
    {
        visitor.on_attribute("turing_model", m_turing_model);
        visitor.on_attribute("model_version", m_model_version);
        visitor.on_attribute("rev", m_rev);
        visitor.on_attribute("serial_number", m_serial_number);
        visitor.on_attribute("enable_turbo", m_enable_turbo);
        visitor.on_attribute("hyper_parameters", m_hyper_parameters);
        visitor.on_attribute("ultra_parameters", m_ultra_parameters);
        return true;
    }

protected:
    TuringModel m_turing_model;
    uint64_t m_model_version;
    int8_t m_rev;
    string m_serial_number;
    bool m_enable_turbo;
    vector<uint64_t> m_hyper_parameters;
    vector<int64_t> m_ultra_parameters;
};

constexpr NodeTypeInfo Oracle::type_info;

class NodeSaver : public AttributeVisitor
{
public:
    NodeSaver(shared_ptr<Node> node)
        : m_node_type_info(node->get_type_info())
    {
        node->visit_attributes(*this);
    }
    const NodeTypeInfo& get_node_type_info() { return m_node_type_info; }
    string& get_string(const string& name) { return m_strings.at(name); }
    bool get_bool(const string& name) { return m_bools.at(name); }
    double get_double(const string& name) { return m_doubles.at(name); }
    int64_t get_signed(const string& name) { return m_signeds.at(name); }
    uint64_t get_unsigned(const string& name) { return m_unsigneds.at(name); }
    vector<float>& get_float_vector(const string& name) { return m_float_vectors.at(name); }
    vector<int64_t>& get_signed_vector(const string& name) { return m_signed_vectors.at(name); }
    vector<string>& get_string_vector(const string& name) { return m_string_vectors.at(name); }
    void set_string(const string& name, const string& value) { m_strings[name] = value; }
    void set_bool(const string& name, bool value) { m_bools[name] = value; }
    void set_double(const string& name, double value) { m_doubles[name] = value; }
    void set_signed(const string& name, int64_t value) { m_signeds[name] = value; }
    void set_unsigned(const string& name, uint64_t value) { m_unsigneds[name] = value; }
    void set_float_vector(const string& name, const vector<float>& value)
    {
        m_float_vectors[name] = value;
    }
    void set_signed_vector(const string& name, const vector<int64_t>& value)
    {
        m_signed_vectors[name] = value;
    }
    void set_string_vector(const string& name, const vector<string>& value)
    {
        m_string_vectors[name] = value;
    }

    void on_attribute(const string& name, string& value) override { set_string(name, value); };
    void on_attribute(const string& name, bool& value) override { set_bool(name, value); }
    void on_adapter(const string& name, ValueAccessor<void>& adapter) override
    {
        NGRAPH_CHECK(false, "name cannot be marshalled");
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const string& name, ValueAccessor<string>& adapter) override
    {
        set_string(name, adapter.get());
    };
    void on_adapter(const string& name, ValueAccessor<int64_t>& adapter) override
    {
        set_signed(name, adapter.get());
    }
    void on_adapter(const string& name, ValueAccessor<double>& adapter) override
    {
        set_double(name, adapter.get());
    }
    void on_adapter(const string& name, ValueAccessor<vector<float>>& adapter) override
    {
        set_float_vector(name, adapter.get());
    }
    void on_adapter(const string& name, ValueAccessor<vector<int64_t>>& adapter) override
    {
        set_signed_vector(name, adapter.get());
    }
    void on_adapter(const string& name, ValueAccessor<vector<string>>& adapter) override
    {
        set_string_vector(name, adapter.get());
    }

protected:
    NodeTypeInfo m_node_type_info;
    map<string, string> m_strings;
    map<string, bool> m_bools;
    map<string, double> m_doubles;
    map<string, int64_t> m_signeds;
    map<string, uint64_t> m_unsigneds;
    map<string, vector<int64_t>> m_signed_vectors;
    map<string, vector<float>> m_float_vectors;
    map<string, vector<std::string>> m_string_vectors;
};

class NodeBuilder : public AttributeVisitor
{
public:
    NodeBuilder(const shared_ptr<Node>& node)
        : m_values(node)
    {
    }

    shared_ptr<Node> create()
    {
        shared_ptr<Node> node(FactoryRegistry<Node>::get().create(m_values.get_node_type_info()));
        node->visit_attributes(*this);
        return node;
    }

    void on_attribute(const string& name, string& value) override
    {
        value = m_values.get_string(name);
    };
    void on_attribute(const string& name, bool& value) override { value = m_values.get_bool(name); }
    void on_adapter(const string& name, ValueAccessor<void>& adapter) override
    {
        NGRAPH_CHECK(false, "name cannot be marshalled");
    }
    // The remaining adapter methods fall back on the void adapter if not implemented
    void on_adapter(const string& name, ValueAccessor<string>& adapter) override
    {
        adapter.set(m_values.get_string(name));
    };
    void on_adapter(const string& name, ValueAccessor<int64_t>& adapter) override
    {
        adapter.set(m_values.get_signed(name));
    }
    void on_adapter(const string& name, ValueAccessor<double>& adapter) override
    {
        adapter.set(m_values.get_double(name));
    }
    void on_adapter(const string& name, ValueAccessor<vector<int64_t>>& adapter) override
    {
        adapter.set(m_values.get_signed_vector(name));
    }
    void on_adapter(const string& name, ValueAccessor<vector<string>>& adapter) override
    {
        adapter.set(m_values.get_string_vector(name));
    }
    void on_adapter(const string& name, ValueAccessor<vector<float>>& adapter) override
    {
        adapter.set(m_values.get_float_vector(name));
    }

protected:
    NodeSaver m_values;
};

TEST(attributes, user_op)
{
    FactoryRegistry<Node>::get().register_factory<Oracle>();
    auto program = make_shared<op::Parameter>(element::i32, Shape{200});
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto oracle = make_shared<Oracle>(program,
                                      data,
                                      TuringModel::XL1200,
                                      2,
                                      4,
                                      "12AU7",
                                      true,
                                      vector<uint64_t>{1, 2, 4, 8},
                                      vector<int64_t>{-1, -2, -4, -8});
    NodeBuilder builder(oracle);
    auto g_oracle = as_type_ptr<Oracle>(builder.create());

    EXPECT_EQ(g_oracle->get_turing_model(), oracle->get_turing_model());
    EXPECT_EQ(g_oracle->get_model_version(), oracle->get_model_version());
    EXPECT_EQ(g_oracle->get_serial_number(), oracle->get_serial_number());
    EXPECT_EQ(g_oracle->get_enable_turbo(), oracle->get_enable_turbo());
    EXPECT_EQ(g_oracle->get_hyper_parameters(), oracle->get_hyper_parameters());
    EXPECT_EQ(g_oracle->get_ultra_parameters(), oracle->get_ultra_parameters());
}

TEST(attributes, reduce_logical_and_op)
{
    // ReduceLogicalAnd derives visit_attributes from op::util::LogicalReductionKeepDims
    FactoryRegistry<Node>::get().register_factory<opset1::ReduceLogicalAnd>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_logical_and = make_shared<opset1::ReduceSum>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_logical_and);
    auto g_reduce_logical_and = as_type_ptr<opset1::ReduceSum>(builder.create());

    EXPECT_EQ(g_reduce_logical_and->get_keep_dims(), reduce_logical_and->get_keep_dims());
}

TEST(attributes, reduce_logical_or_op)
{
    // ReduceLogicalOr derives visit_attributes from op::util::LogicalReductionKeepDims
    FactoryRegistry<Node>::get().register_factory<opset1::ReduceLogicalOr>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_logical_or = make_shared<opset1::ReduceLogicalOr>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_logical_or);
    auto g_reduce_logical_or = as_type_ptr<opset1::ReduceLogicalOr>(builder.create());

    EXPECT_EQ(g_reduce_logical_or->get_keep_dims(), reduce_logical_or->get_keep_dims());
}

TEST(attributes, reduce_max_op)
{
    // ReduceMax derives visit_attributes from op::util::ArithmeticReductionKeepDims
    FactoryRegistry<Node>::get().register_factory<opset1::ReduceMax>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_max = make_shared<opset1::ReduceMax>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_max);
    auto g_reduce_max = as_type_ptr<opset1::ReduceMax>(builder.create());

    EXPECT_EQ(g_reduce_max->get_keep_dims(), reduce_max->get_keep_dims());
}

TEST(attributes, reduce_mean_op)
{
    // ReduceMean derives visit_attributes from op::util::ArithmeticReductionKeepDims
    FactoryRegistry<Node>::get().register_factory<opset1::ReduceMean>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_mean = make_shared<opset1::ReduceMean>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_mean);
    auto g_reduce_mean = as_type_ptr<opset1::ReduceMean>(builder.create());

    EXPECT_EQ(g_reduce_mean->get_keep_dims(), reduce_mean->get_keep_dims());
}

TEST(attributes, reduce_min_op)
{
    // ReduceMax derives visit_attributes from op::util::ArithmeticReductionKeepDims
    FactoryRegistry<Node>::get().register_factory<opset1::ReduceMin>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_min = make_shared<opset1::ReduceMin>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_min);
    auto g_reduce_min = as_type_ptr<opset1::ReduceMin>(builder.create());

    EXPECT_EQ(g_reduce_min->get_keep_dims(), reduce_min->get_keep_dims());
}

TEST(attributes, reduce_prod_op)
{
    // ReduceProd derives visit_attributes from op::util::ArithmeticReductionKeepDims
    FactoryRegistry<Node>::get().register_factory<opset1::ReduceProd>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_prod = make_shared<opset1::ReduceProd>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_prod);
    auto g_reduce_prod = as_type_ptr<opset1::ReduceProd>(builder.create());

    EXPECT_EQ(g_reduce_prod->get_keep_dims(), reduce_prod->get_keep_dims());
}

TEST(attributes, reduce_sum_op)
{
    // ReduceSum derives visit_attributes from op::util::ArithmeticReductionKeepDims
    FactoryRegistry<Node>::get().register_factory<opset1::ReduceSum>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{3, 4, 5});
    auto reduction_axes = make_shared<op::Parameter>(element::i64, Shape{2});

    bool keep_dims = true;

    auto reduce_sum = make_shared<opset1::ReduceSum>(data, reduction_axes, keep_dims);
    NodeBuilder builder(reduce_sum);
    auto g_reduce_sum = as_type_ptr<opset1::ReduceSum>(builder.create());

    EXPECT_EQ(g_reduce_sum->get_keep_dims(), reduce_sum->get_keep_dims());
}

TEST(attributes, region_yolo_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::RegionYolo>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});

    size_t num_coords = 1;
    size_t num_classes = 1;
    size_t num_regions = 1;
    auto do_softmax = false;
    auto mask = std::vector<int64_t>{0, 0};
    auto axis = 1;
    auto end_axis = 2;
    auto anchors = std::vector<float>{1};

    auto region_yolo = make_shared<opset1::RegionYolo>(
        data, num_coords, num_classes, num_regions, do_softmax, mask, axis, end_axis, anchors);
    NodeBuilder builder(region_yolo);
    auto g_region_yolo = as_type_ptr<opset1::RegionYolo>(builder.create());

    EXPECT_EQ(g_region_yolo->get_num_coords(), region_yolo->get_num_coords());
    EXPECT_EQ(g_region_yolo->get_num_classes(), region_yolo->get_num_classes());
    EXPECT_EQ(g_region_yolo->get_num_regions(), region_yolo->get_num_regions());
    EXPECT_EQ(g_region_yolo->get_do_softmax(), region_yolo->get_do_softmax());
    EXPECT_EQ(g_region_yolo->get_mask(), region_yolo->get_mask());
    EXPECT_EQ(g_region_yolo->get_anchors(), region_yolo->get_anchors());
    EXPECT_EQ(g_region_yolo->get_axis(), region_yolo->get_axis());
    EXPECT_EQ(g_region_yolo->get_end_axis(), region_yolo->get_end_axis());
}

TEST(attributes, reshape_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Reshape>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4});
    auto pattern = make_shared<op::Parameter>(element::i32, Shape{2});

    bool special_zero = true;

    auto reshape = make_shared<opset1::Reshape>(data, pattern, special_zero);
    NodeBuilder builder(reshape);
    auto g_reshape = as_type_ptr<opset1::Reshape>(builder.create());

    EXPECT_EQ(g_reshape->get_special_zero(), reshape->get_special_zero());
}

TEST(attributes, reverse_op_enum_mode)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Reverse>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<op::Parameter>(element::i32, Shape{200});

    auto reverse = make_shared<opset1::Reverse>(data, reversed_axes, opset1::Reverse::Mode::INDEX);
    NodeBuilder builder(reverse);
    auto g_reverse = as_type_ptr<opset1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}

TEST(attributes, reverse_op_string_mode)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Reverse>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto reversed_axes = make_shared<op::Parameter>(element::i32, Shape{200});

    std::string mode = "index";

    auto reverse = make_shared<opset1::Reverse>(data, reversed_axes, mode);
    NodeBuilder builder(reverse);
    auto g_reverse = as_type_ptr<opset1::Reverse>(builder.create());

    EXPECT_EQ(g_reverse->get_mode(), reverse->get_mode());
}

TEST(attributes, reverse_sequence_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::ReverseSequence>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 2});
    auto seq_indices = make_shared<op::Parameter>(element::i32, Shape{4});

    auto batch_axis = 2;
    auto seq_axis = 1;

    auto reverse_sequence =
        make_shared<opset1::ReverseSequence>(data, seq_indices, batch_axis, seq_axis);
    NodeBuilder builder(reverse_sequence);
    auto g_reverse_sequence = as_type_ptr<opset1::ReverseSequence>(builder.create());

    EXPECT_EQ(g_reverse_sequence->get_origin_batch_axis(),
              reverse_sequence->get_origin_batch_axis());
    EXPECT_EQ(g_reverse_sequence->get_origin_sequence_axis(),
              reverse_sequence->get_origin_sequence_axis());
}

TEST(attributes, rnn_cell_op_custom_attributes)
{
    FactoryRegistry<Node>::get().register_factory<opset1::RNNCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;
    auto activations = std::vector<std::string>{"sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    float clip = 1.0;

    auto rnn_cell = make_shared<opset1::RNNCell>(X, H_t, W, R, hidden_size);

    NodeBuilder builder(rnn_cell);
    auto g_rnn_cell = as_type_ptr<opset1::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}

TEST(attributes, rnn_cell_op_default_attributes)
{
    FactoryRegistry<Node>::get().register_factory<opset1::RNNCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H_t = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;

    auto rnn_cell = make_shared<opset1::RNNCell>(X, H_t, W, R, hidden_size);

    NodeBuilder builder(rnn_cell);
    auto g_rnn_cell = as_type_ptr<opset1::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
}
