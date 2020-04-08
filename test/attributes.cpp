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
    shared_ptr<Node> clone_with_new_inputs(const OutputVector& args) const override
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
    // ReduceMin derives visit_attributes from op::util::ArithmeticReductionKeepDims
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
    auto data = make_shared<op::Parameter>(element::i64, Shape{1, 255, 26, 26});

    size_t num_coords = 4;
    size_t num_classes = 1;
    size_t num_regions = 6;
    auto do_softmax = false;
    auto mask = std::vector<int64_t>{0, 1};
    auto axis = 1;
    auto end_axis = 3;
    auto anchors = std::vector<float>{10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};

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
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;
    auto activations = std::vector<std::string>{"sigmoid", "tanh"};
    auto activations_alpha = std::vector<float>{1.0, 1.5};
    auto activations_beta = std::vector<float>{2.0, 1.0};
    float clip = 1.0;

    auto rnn_cell = make_shared<opset1::RNNCell>(
        X, H, W, R, hidden_size, activations, activations_alpha, activations_beta, clip);

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
    auto H = make_shared<op::Parameter>(element::f32, Shape{2, 3});
    auto W = make_shared<op::Parameter>(element::f32, Shape{3, 3});
    auto R = make_shared<op::Parameter>(element::f32, Shape{3, 3});

    const size_t hidden_size = 3;

    auto rnn_cell = make_shared<opset1::RNNCell>(X, H, W, R, hidden_size);

    NodeBuilder builder(rnn_cell);
    auto g_rnn_cell = as_type_ptr<opset1::RNNCell>(builder.create());

    EXPECT_EQ(g_rnn_cell->get_hidden_size(), rnn_cell->get_hidden_size());
    EXPECT_EQ(g_rnn_cell->get_clip(), rnn_cell->get_clip());
    EXPECT_EQ(g_rnn_cell->get_activations(), rnn_cell->get_activations());
    EXPECT_EQ(g_rnn_cell->get_activations_alpha(), rnn_cell->get_activations_alpha());
    EXPECT_EQ(g_rnn_cell->get_activations_beta(), rnn_cell->get_activations_beta());
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
