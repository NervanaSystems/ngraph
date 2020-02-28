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

TEST(attributes, matmul_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::MatMul>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{0, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 0});

    bool transpose_a = true;
    bool transpose_b = true;

    auto matmul = make_shared<opset1::MatMul>(A, B, transpose_a, transpose_b);
    NodeBuilder builder(matmul);
    auto g_matmul = as_type_ptr<opset1::MatMul>(builder.create());

    EXPECT_EQ(g_matmul->get_transpose_a(), matmul->get_transpose_a());
    EXPECT_EQ(g_matmul->get_transpose_b(), matmul->get_transpose_b());
}

TEST(attributes, max_pool_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::MaxPool>();
    auto data = make_shared<op::Parameter>(element::f32, Shape{64, 3, 5});

    auto strides = Strides{2};
    auto pads_begin = Shape{1};
    auto pads_end = Shape{1};
    auto kernel = Shape{1};
    auto rounding_mode = op::RoundingType::FLOOR;
    auto auto_pad = op::PadType::EXPLICIT;

    auto max_pool = make_shared<opset1::MaxPool>(
        data, strides, pads_begin, pads_end, kernel, rounding_mode, auto_pad);
    NodeBuilder builder(max_pool);
    auto g_max_pool = as_type_ptr<opset1::MaxPool>(builder.create());

    EXPECT_EQ(g_max_pool->get_strides(), max_pool->get_strides());
    EXPECT_EQ(g_max_pool->get_pads_begin(), max_pool->get_pads_begin());
    EXPECT_EQ(g_max_pool->get_pads_end(), max_pool->get_pads_end());
    EXPECT_EQ(g_max_pool->get_kernel(), max_pool->get_kernel());
    EXPECT_EQ(g_max_pool->get_rounding_type(), max_pool->get_rounding_type());
    EXPECT_EQ(g_max_pool->get_auto_pad(), max_pool->get_auto_pad());
}

TEST(attributes, mod_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Mod>();
    auto A = make_shared<op::Parameter>(element::f32, Shape{0, 2});
    auto B = make_shared<op::Parameter>(element::f32, Shape{2, 0});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto mod = make_shared<opset1::Mod>(A, B, auto_broadcast);
    NodeBuilder builder(mod);
    auto g_mod = as_type_ptr<opset1::Mod>(builder.create());

    EXPECT_EQ(g_mod->get_auto_broadcast(), mod->get_auto_broadcast());
}

TEST(attributes, non_max_suppression_op_custom_attributes)
{
    FactoryRegistry<Node>::get().register_factory<opset1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto box_encoding = opset1::NonMaxSuppression::BoxEncodingType::CENTER;
    bool sort_result_descending = false;

    auto nms =
        make_shared<opset1::NonMaxSuppression>(boxes, scores, box_encoding, sort_result_descending);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, non_max_suppression_op_default_attributes)
{
    FactoryRegistry<Node>::get().register_factory<opset1::NonMaxSuppression>();
    auto boxes = make_shared<op::Parameter>(element::f32, Shape{1, 1, 4});
    auto scores = make_shared<op::Parameter>(element::f32, Shape{1, 1, 1});

    auto nms = make_shared<opset1::NonMaxSuppression>(boxes, scores);
    NodeBuilder builder(nms);
    auto g_nms = as_type_ptr<opset1::NonMaxSuppression>(builder.create());

    EXPECT_EQ(g_nms->get_box_encoding(), nms->get_box_encoding());
    EXPECT_EQ(g_nms->get_sort_result_descending(), nms->get_sort_result_descending());
}

TEST(attributes, normalize_l2_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::NormalizeL2>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{1});
    const auto axes = make_shared<op::Constant>(element::i32, Shape{}, vector<int32_t>{0});

    float eps{1e-6f};
    auto eps_mode = op::EpsMode::ADD;

    auto normalize_l2 = make_shared<opset1::NormalizeL2>(data, axes, eps, eps_mode);
    NodeBuilder builder(normalize_l2);
    auto g_normalize_l2 = as_type_ptr<opset1::NormalizeL2>(builder.create());

    EXPECT_EQ(g_normalize_l2->get_eps(), normalize_l2->get_eps());
    EXPECT_EQ(g_normalize_l2->get_eps_mode(), normalize_l2->get_eps_mode());
}

TEST(attributes, one_hot_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::OneHot>();
    auto indices = make_shared<op::Parameter>(element::i64, Shape{1, 3, 2, 3});
    auto depth = op::Constant::create(element::i64, Shape{}, {4});
    auto on_value = op::Constant::create(element::f32, Shape{}, {1.0f});
    auto off_value = op::Constant::create(element::f32, Shape{}, {0.0f});

    int64_t axis = 3;

    auto one_hot = make_shared<opset1::OneHot>(indices, depth, on_value, off_value, axis);
    NodeBuilder builder(one_hot);
    auto g_one_hot = as_type_ptr<opset1::OneHot>(builder.create());

    EXPECT_EQ(g_one_hot->get_axis(), one_hot->get_axis());
}

TEST(attributes, pad_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Pad>();
    auto arg = make_shared<op::Parameter>(element::f32, Shape{1, 2, 3});
    auto pads_begin = make_shared<op::Parameter>(element::i64, Shape{1});
    auto pads_end = make_shared<op::Parameter>(element::i64, Shape{1});

    auto pad_mode = op::PadMode::EDGE;

    auto pad = make_shared<opset1::Pad>(arg, pads_begin, pads_end, pad_mode);
    NodeBuilder builder(pad);
    auto g_pad = as_type_ptr<opset1::Pad>(builder.create());

    EXPECT_EQ(g_pad->get_pad_mode(), pad->get_pad_mode());
}

TEST(attributes, psroi_pooling_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::PSROIPooling>();
    auto input = make_shared<op::Parameter>(element::f32, Shape{1, 1024, 63, 38});
    auto coords = make_shared<op::Parameter>(element::f32, Shape{300, 5});

    const int64_t output_dim = 882;
    const int64_t group_size = 3;
    const float spatial_scale = 0.0625;
    int spatial_bins_x = 1;
    int spatial_bins_y = 1;
    string mode = "Avg";

    auto psroi_pool = make_shared<opset1::PSROIPooling>(
        input, coords, output_dim, group_size, spatial_scale, spatial_bins_x, spatial_bins_y, mode);
    NodeBuilder builder(psroi_pool);
    auto g_psroi_pool = as_type_ptr<opset1::PSROIPooling>(builder.create());

    EXPECT_EQ(g_psroi_pool->get_output_dim(), psroi_pool->get_output_dim());
    EXPECT_EQ(g_psroi_pool->get_group_size(), psroi_pool->get_group_size());
    EXPECT_EQ(g_psroi_pool->get_spatial_scale(), psroi_pool->get_spatial_scale());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_x(), psroi_pool->get_spatial_bins_x());
    EXPECT_EQ(g_psroi_pool->get_spatial_bins_y(), psroi_pool->get_spatial_bins_y());
    EXPECT_EQ(g_psroi_pool->get_mode(), psroi_pool->get_mode());
}
