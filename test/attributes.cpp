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
    vector<int64_t>& get_signed_vector(const string& name) { return m_signed_vectors.at(name); }
    void set_string(const string& name, const string& value) { m_strings[name] = value; }
    void set_bool(const string& name, bool value) { m_bools[name] = value; }
    void set_double(const string& name, double value) { m_doubles[name] = value; }
    void set_signed(const string& name, int64_t value) { m_signeds[name] = value; }
    void set_unsigned(const string& name, uint64_t value) { m_unsigneds[name] = value; }
    void set_signed_vector(const string& name, const vector<int64_t>& value)
    {
        m_signed_vectors[name] = value;
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
    void on_adapter(const string& name, ValueAccessor<vector<int64_t>>& adapter) override
    {
        set_signed_vector(name, adapter.get());
    }
    void on_adapter(const string& name, ValueAccessor<int64_t>& adapter) override
    {
        set_signed(name, adapter.get());
    }
    void on_adapter(const string& name, ValueAccessor<double>& adapter) override
    {
        set_double(name, adapter.get());
    }

protected:
    NodeTypeInfo m_node_type_info;
    map<string, string> m_strings;
    map<string, bool> m_bools;
    map<string, double> m_doubles;
    map<string, int64_t> m_signeds;
    map<string, uint64_t> m_unsigneds;
    map<string, vector<int64_t>> m_signed_vectors;
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
    void on_adapter(const string& name, ValueAccessor<vector<int64_t>>& adapter) override
    {
        adapter.set(m_values.get_signed_vector(name));
    }
    void on_adapter(const string& name, ValueAccessor<int64_t>& adapter) override
    {
        adapter.set(m_values.get_signed(name));
    }
    void on_adapter(const string& name, ValueAccessor<double>& adapter) override
    {
        adapter.set(m_values.get_double(name));
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

TEST(attributes, shuffle_channels_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::ShuffleChannels>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto groups = 2;
    auto shuffle_channels = make_shared<opset1::ShuffleChannels>(data, axis, groups);
    NodeBuilder builder(shuffle_channels);
    auto g_shuffle_channels = as_type_ptr<opset1::ShuffleChannels>(builder.create());

    EXPECT_EQ(g_shuffle_channels->get_axis(), shuffle_channels->get_axis());
    EXPECT_EQ(g_shuffle_channels->get_groups(), shuffle_channels->get_groups());
}

TEST(attributes, softmax_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Softmax>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = 0;
    auto softmax = make_shared<opset1::Softmax>(data, axis);
    NodeBuilder builder(softmax);
    auto g_softmax = as_type_ptr<opset1::Softmax>(builder.create());

    EXPECT_EQ(g_softmax->get_axis(), softmax->get_axis());
}

TEST(attributes, space_to_depth_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::SpaceToDepth>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 50, 50});
    auto block_size = 2;
    auto mode = opset1::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    auto space_to_depth = make_shared<opset1::SpaceToDepth>(data, mode, block_size);
    NodeBuilder builder(space_to_depth);
    auto g_space_to_depth = as_type_ptr<opset1::SpaceToDepth>(builder.create());

    EXPECT_EQ(g_space_to_depth->get_block_size(), space_to_depth->get_block_size());
    EXPECT_EQ(g_space_to_depth->get_mode(), space_to_depth->get_mode());
}

TEST(attributes, split_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::Split>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{200});
    auto axis = make_shared<op::Parameter>(element::i32, Shape{});
    auto num_splits = 2;
    auto split = make_shared<opset1::Split>(data, axis, num_splits);
    NodeBuilder builder(split);
    auto g_split = as_type_ptr<opset1::Split>(builder.create());

    EXPECT_EQ(g_split->get_num_splits(), split->get_num_splits());
}

TEST(attributes, squared_difference_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::SquaredDifference>();
    auto x1 = make_shared<op::Parameter>(element::i32, Shape{200});
    auto x2 = make_shared<op::Parameter>(element::i32, Shape{200});
    auto auto_broadcast = op::AutoBroadcastType::NUMPY;
    auto squared_difference = make_shared<opset1::SquaredDifference>(x1, x2, auto_broadcast);
    NodeBuilder builder(squared_difference);
    auto g_squared_difference = as_type_ptr<opset1::SquaredDifference>(builder.create());

    EXPECT_EQ(g_squared_difference->get_autob(), squared_difference->get_autob());
}

TEST(attributes, strided_slice_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::StridedSlice>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto begin = make_shared<op::Parameter>(element::i32, Shape{2});
    auto end = make_shared<op::Parameter>(element::i32, Shape{2});
    auto stride = make_shared<op::Parameter>(element::i32, Shape{2});

    auto begin_mask = std::vector<int64_t>{0, 0};
    auto end_mask = std::vector<int64_t>{0, 0};
    auto new_axis_mask = std::vector<int64_t>{0, 0};
    auto shrink_axis_mask = std::vector<int64_t>{0, 0};
    auto ellipsis_mask = std::vector<int64_t>{0, 0};

    auto strided_slice = make_shared<opset1::StridedSlice>(data,
                                                           begin,
                                                           end,
                                                           stride,
                                                           begin_mask,
                                                           end_mask,
                                                           new_axis_mask,
                                                           shrink_axis_mask,
                                                           ellipsis_mask);
    NodeBuilder builder(strided_slice);
    auto g_strided_slice = as_type_ptr<opset1::StridedSlice>(builder.create());

    EXPECT_EQ(g_strided_slice->get_begin_mask(), strided_slice->get_begin_mask());
    EXPECT_EQ(g_strided_slice->get_end_mask(), strided_slice->get_end_mask());
    EXPECT_EQ(g_strided_slice->get_new_axis_mask(), strided_slice->get_new_axis_mask());
    EXPECT_EQ(g_strided_slice->get_shrink_axis_mask(), strided_slice->get_shrink_axis_mask());
    EXPECT_EQ(g_strided_slice->get_ellipsis_mask(), strided_slice->get_ellipsis_mask());
}

TEST(attributes, topk_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::TopK>();
    auto data = make_shared<op::Parameter>(element::i32, Shape{2, 3, 4, 5});
    auto k = make_shared<op::Parameter>(element::i32, Shape{});

    auto axis = 0;
    auto mode = opset1::TopK::Mode::MAX;
    auto sort_type = opset1::TopK::SortType::SORT_VALUES;

    auto topk = make_shared<opset1::TopK>(data, k, axis, mode, sort_type);
    NodeBuilder builder(topk);
    auto g_topk = as_type_ptr<opset1::TopK>(builder.create());

    EXPECT_EQ(g_topk->get_axis(), topk->get_axis());
    EXPECT_EQ(g_topk->get_mode(), topk->get_mode());
    EXPECT_EQ(g_topk->get_sort_type(), topk->get_sort_type());
}

TEST(attributes, logical_xor_op)
{
    FactoryRegistry<Node>::get().register_factory<opset1::LogicalXor>();
    auto x1 = make_shared<op::Parameter>(element::boolean, Shape{200});
    auto x2 = make_shared<op::Parameter>(element::boolean, Shape{200});

    auto auto_broadcast = op::AutoBroadcastType::NUMPY;

    auto logical_xor = make_shared<opset1::LogicalXor>(x1, x2, auto_broadcast);
    NodeBuilder builder(logical_xor);
    auto g_logical_xor = as_type_ptr<opset1::LogicalXor>(builder.create());

    EXPECT_EQ(g_logical_xor->get_autob(), logical_xor->get_autob());
}
