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

TEST(attributes, elu_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Elu>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 4});

    double alpha = 0.1;

    const auto elu = make_shared<opset1::Elu>(data, alpha);
    NodeBuilder builder(elu);
    auto g_elu = as_type_ptr<opset1::Elu>(builder.create());

    EXPECT_EQ(g_elu->get_alpha(), elu->get_alpha());
}

TEST(attributes, fake_quantize_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::FakeQuantize>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto input_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto input_high = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_low = make_shared<op::Parameter>(element::f32, Shape{});
    const auto output_high = make_shared<op::Parameter>(element::f32, Shape{});

    auto levels = 5;
    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    const auto fake_quantize = make_shared<op::FakeQuantize>(
        data, input_low, input_high, output_low, output_high, levels, auto_broadcast);
    NodeBuilder builder(fake_quantize);
    auto g_fake_quantize = as_type_ptr<opset1::FakeQuantize>(builder.create());

    EXPECT_EQ(g_fake_quantize->get_levels(), fake_quantize->get_levels());
    EXPECT_EQ(g_fake_quantize->get_auto_broadcast(), fake_quantize->get_auto_broadcast());
}

TEST(attributes, grn_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::GRN>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 4, 5});

    float bias = 1.25f;

    auto grn = make_shared<opset1::GRN>(data, bias);
    NodeBuilder builder(grn);
    auto g_grn = as_type_ptr<opset1::GRN>(builder.create());

    EXPECT_EQ(g_grn->get_bias(), grn->get_bias());
}

TEST(attributes, group_conv_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::GroupConvolution>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 12, 224, 224});
    auto filters = make_shared<op::Parameter>(element::f32, Shape{4, 1, 3, 5, 5});
    auto strides = Strides{1, 1};
    auto pads_begin = CoordinateDiff{1, 2};
    auto pads_end = CoordinateDiff{1, 2};
    auto dilations = Strides{1, 1};
    auto group_conv = make_shared<opset1::GroupConvolution>(
        data, filters, strides, pads_begin, pads_end, dilations, op::PadType::VALID);
    NodeBuilder builder(group_conv);
    auto g_group_conv = as_type_ptr<opset1::GroupConvolution>(builder.create());
    EXPECT_EQ(g_group_conv->get_strides(), group_conv->get_strides());
    EXPECT_EQ(g_group_conv->get_pads_begin(), group_conv->get_pads_begin());
    EXPECT_EQ(g_group_conv->get_pads_end(), group_conv->get_pads_end());
    EXPECT_EQ(g_group_conv->get_dilations(), group_conv->get_dilations());
    EXPECT_EQ(g_group_conv->get_auto_pad(), group_conv->get_auto_pad());
}

TEST(attributes, group_conv_backprop_data_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::GroupConvolutionBackpropData>();
    const auto data = make_shared<op::Parameter>(element::f32, Shape{1, 20, 224, 224});
    const auto filter = make_shared<op::Parameter>(element::f32, Shape{4, 5, 2, 3, 3});
    const auto output_shape = make_shared<op::Parameter>(element::f32, Shape{1, 8, 447, 447});

    const auto strides = Strides{2, 1};
    const auto pads_begin = CoordinateDiff{3, 4};
    const auto pads_end = CoordinateDiff{4, 6};
    const auto dilations = Strides{3, 1};
    const auto auto_pad = op::PadType::EXPLICIT;
    const auto output_padding = CoordinateDiff{3, 4};

    const auto gcbd = make_shared<opset1::GroupConvolutionBackpropData>(data,
                                                                        filter,
                                                                        output_shape,
                                                                        strides,
                                                                        pads_begin,
                                                                        pads_end,
                                                                        dilations,
                                                                        auto_pad,
                                                                        output_padding);
    NodeBuilder builder(gcbd);
    const auto g_gcbd = as_type_ptr<opset1::GroupConvolutionBackpropData>(builder.create());

    EXPECT_EQ(g_gcbd->get_strides(), gcbd->get_strides());
    EXPECT_EQ(g_gcbd->get_pads_begin(), gcbd->get_pads_begin());
    EXPECT_EQ(g_gcbd->get_pads_end(), gcbd->get_pads_end());
    EXPECT_EQ(g_gcbd->get_dilations(), gcbd->get_dilations());
    EXPECT_EQ(g_gcbd->get_auto_pad(), gcbd->get_auto_pad());
    EXPECT_EQ(g_gcbd->get_output_padding(), gcbd->get_output_padding());
}

TEST(attributes, lrn_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::LRN>();
    const auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3, 4});
    const auto axes = make_shared<op::Parameter>(element::i32, Shape{2});

    const double alpha = 1.1;
    const double beta = 2.2;
    const double bias = 3.3;
    const size_t size = 4;

    const auto lrn = make_shared<opset1::LRN>(arg, axes, alpha, beta, bias, size);
    NodeBuilder builder(lrn);
    auto g_lrn = as_type_ptr<opset1::LRN>(builder.create());

    EXPECT_EQ(g_lrn->get_alpha(), lrn->get_alpha());
    EXPECT_EQ(g_lrn->get_beta(), lrn->get_beta());
    EXPECT_EQ(g_lrn->get_bias(), lrn->get_bias());
    EXPECT_EQ(g_lrn->get_nsize(), lrn->get_nsize());
}

TEST(attributes, lstm_cell_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::LSTMCell>();
    auto X = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{12, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{12, 3});
    const auto initial_hidden_state = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    const auto initial_cell_state = make_shared<op::Parameter>(element::f32, Shape{2, 3});

    const auto hidden_size = 3;
    const auto weights_format = op::LSTMWeightsFormat::ICOF;
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    const float clip = 0.5f;
    bool input_forget = true;

    const auto lstm_cell = make_shared<opset1::LSTMCell>(X,
                                                         initial_hidden_state,
                                                         initial_cell_state,
                                                         W,
                                                         R,
                                                         hidden_size,
                                                         weights_format,
                                                         activations,
                                                         activations_alpha,
                                                         activations_beta,
                                                         clip,
                                                         input_forget);
    NodeBuilder builder(lstm_cell);
    auto g_lstm_cell = as_type_ptr<opset1::LSTMCell>(builder.create());

    EXPECT_EQ(g_lstm_cell->get_hidden_size(), lstm_cell->get_hidden_size());
    EXPECT_EQ(g_lstm_cell->get_activations(), lstm_cell->get_activations());
    EXPECT_EQ(g_lstm_cell->get_activations_alpha(), lstm_cell->get_activations_alpha());
    EXPECT_EQ(g_lstm_cell->get_activations_beta(), lstm_cell->get_activations_beta());
    EXPECT_EQ(g_lstm_cell->get_clip(), lstm_cell->get_clip());
    EXPECT_EQ(g_lstm_cell->get_input_forget(), lstm_cell->get_input_forget());
    EXPECT_EQ(g_lstm_cell->get_weights_format(), lstm_cell->get_weights_format());
}

TEST(attributes, lstm_sequence_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::LSTMSequence>();
    const auto X = make_shared<op::Parameter>(element::f32, Shape{1, 2, 4});
    const auto initial_hidden_state = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto initial_cell_state = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    const auto sequence_lengths = make_shared<op::Parameter>(element::i32, Shape{2});
    const auto W = make_shared<op::Parameter>(element::f32, Shape{1, 12, 4});
    const auto R = make_shared<op::Parameter>(element::f32, Shape{1, 12, 3});
    const auto B = make_shared<op::Parameter>(element::f32, Shape{1, 12});

    const auto hidden_size = 3;
    const auto lstm_direction = op::LSTMSequence::direction::FORWARD;
    const auto weights_format = op::LSTMWeightsFormat::ICOF;
    const std::vector<float> activations_alpha = {1, 2, 3};
    const std::vector<float> activations_beta = {4, 5, 6};
    const std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};
    const float clip_threshold = 0.5f;
    const bool input_forget = true;

    const auto lstm_sequence = make_shared<opset1::LSTMSequence>(X,
                                                                 initial_hidden_state,
                                                                 initial_cell_state,
                                                                 sequence_lengths,
                                                                 W,
                                                                 R,
                                                                 B,
                                                                 hidden_size,
                                                                 lstm_direction,
                                                                 weights_format,
                                                                 activations_alpha,
                                                                 activations_beta,
                                                                 activations,
                                                                 clip_threshold,
                                                                 input_forget);
    NodeBuilder builder(lstm_sequence);
    auto g_lstm_sequence = as_type_ptr<opset1::LSTMSequence>(builder.create());

    EXPECT_EQ(g_lstm_sequence->get_hidden_size(), lstm_sequence->get_hidden_size());
    EXPECT_EQ(g_lstm_sequence->get_activations(), lstm_sequence->get_activations());
    EXPECT_EQ(g_lstm_sequence->get_activations_alpha(), lstm_sequence->get_activations_alpha());
    EXPECT_EQ(g_lstm_sequence->get_activations_beta(), lstm_sequence->get_activations_beta());
    EXPECT_EQ(g_lstm_sequence->get_clip_threshold(), lstm_sequence->get_clip_threshold());
    EXPECT_EQ(g_lstm_sequence->get_direction(), lstm_sequence->get_direction());
    EXPECT_EQ(g_lstm_sequence->get_input_forget(), lstm_sequence->get_input_forget());
    EXPECT_EQ(g_lstm_sequence->get_weights_format(), lstm_sequence->get_weights_format());
}
