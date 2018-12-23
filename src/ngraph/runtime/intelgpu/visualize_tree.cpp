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
#include <map>
#include <memory>

#include "ngraph/runtime/intelgpu/visualize_tree.hpp"

#include "ngraph/node.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reduce_window.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/op_tbl.hpp"
    UNDEFINED_OP
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
    auto it = typeid_map.find(s);
    if (it == typeid_map.end())
    {
        return OP_TYPEID::UNDEFINED_OP;
    }
    return it->second;
}

static const string table_begin = "\n<table border=\"0\">";
static const string table_end = "\n</table>";
static const string cell_end = "</td>";
static const string table_row_end = cell_end + "</tr>";
static const string font_small_begin = "<font point-size=\"7\">";
static const string font_end = "</font>";

static string cell_begin(const string& align = string("left"))
{
    return string("<td align=\"") + align + "\">";
}

static string table_row_begin(const string& align = string("left"))
{
    return string("\n<tr>") + cell_begin(align);
}

template <typename T>
static string print_table_row_dims(const string& name, const T& shape)
{
    return table_row_begin() + font_small_begin + name + vector_to_string(shape) + font_end +
           table_row_end;
}

template <typename T>
static string print_table_row_value(const string& name, T val)
{
    stringstream result;

    result << table_row_begin() << font_small_begin << name << ":" << val << font_end
           << table_row_end;

    return result.str();
}

void print_node_parameters(ostringstream& writer, const shared_ptr<Node>& node)
{
    switch (get_typeid(node->description()))
    {
    case OP_TYPEID::BatchNormTrainingBackprop:
    {
        const shared_ptr<op::BatchNormTrainingBackprop> batch_norm =
            static_pointer_cast<op::BatchNormTrainingBackprop>(node);

        writer << print_table_row_value("EPS", batch_norm->get_eps_value());
        break;
    }
    case OP_TYPEID::BatchNormInference:
    {
        const shared_ptr<op::BatchNormInference> batch_norm =
            static_pointer_cast<op::BatchNormInference>(node);

        writer << print_table_row_value("EPS", batch_norm->get_eps_value());
        break;
    }
    case OP_TYPEID::BatchNormTraining:
    {
        const shared_ptr<op::BatchNormTraining> batch_norm =
            static_pointer_cast<op::BatchNormTraining>(node);

        writer << print_table_row_value("EPS", batch_norm->get_eps_value());
        break;
    }
    case OP_TYPEID::GetOutputElement:
    {
        const shared_ptr<op::GetOutputElement> elem =
            static_pointer_cast<op::GetOutputElement>(node);

        writer << print_table_row_value("element", elem->get_n());
        break;
    }
    case OP_TYPEID::MaxPool:
    {
        const shared_ptr<op::MaxPool> max_pool = static_pointer_cast<op::MaxPool>(node);

        writer << print_table_row_dims("win_shape", max_pool->get_window_shape())
               << print_table_row_dims("win_strides", max_pool->get_window_movement_strides())
               << print_table_row_dims("pad_above", max_pool->get_padding_above())
               << print_table_row_dims("pad_below", max_pool->get_padding_below());
        break;
    }
    case OP_TYPEID::MaxPoolBackprop:
    {
        const shared_ptr<op::MaxPoolBackprop> max_pool_b =
            static_pointer_cast<op::MaxPoolBackprop>(node);

        writer << print_table_row_dims("win_shape", max_pool_b->get_window_shape())
               << print_table_row_dims("win_strides", max_pool_b->get_window_movement_strides())
               << print_table_row_dims("pad_above", max_pool_b->get_padding_above())
               << print_table_row_dims("pad_below", max_pool_b->get_padding_below());
        break;
    }
    case OP_TYPEID::AvgPool:
    {
        const shared_ptr<op::AvgPool> avg_pool = static_pointer_cast<op::AvgPool>(node);

        writer << print_table_row_dims("win_shape", avg_pool->get_window_shape())
               << print_table_row_dims("win_strides", avg_pool->get_window_movement_strides())
               << print_table_row_dims("pad_above", avg_pool->get_padding_above())
               << print_table_row_dims("pad_below", avg_pool->get_padding_below())
               << print_table_row_value("pad_included",
                                        avg_pool->get_include_padding_in_avg_computation());
        break;
    }
    case OP_TYPEID::AvgPoolBackprop:
    {
        const shared_ptr<op::AvgPoolBackprop> avg_pool_b =
            static_pointer_cast<op::AvgPoolBackprop>(node);

        writer << print_table_row_dims("win_shape", avg_pool_b->get_window_shape())
               << print_table_row_dims("win_strides", avg_pool_b->get_window_movement_strides())
               << print_table_row_dims("pad_above", avg_pool_b->get_padding_above())
               << print_table_row_dims("pad_below", avg_pool_b->get_padding_below())
               << print_table_row_value("pad_included",
                                        avg_pool_b->get_include_padding_in_avg_computation());
        break;
    }
    case OP_TYPEID::Broadcast:
    {
        const shared_ptr<op::Broadcast> broadcast = static_pointer_cast<op::Broadcast>(node);

        writer << print_table_row_dims("broadcast_axis", broadcast->get_broadcast_axes());
        break;
    }
    case OP_TYPEID::Max:
    case OP_TYPEID::Min:
    case OP_TYPEID::Product:
    case OP_TYPEID::Sum:
    {
        const shared_ptr<op::util::ArithmeticReduction> arith_op =
            static_pointer_cast<op::util::ArithmeticReduction>(node);

        writer << print_table_row_dims("reduction_axis", arith_op->get_reduction_axes());
        break;
    }
    case OP_TYPEID::All:
    case OP_TYPEID::Any:
    {
        const shared_ptr<op::util::LogicalReduction> logical_op =
            static_pointer_cast<op::util::LogicalReduction>(node);

        writer << print_table_row_dims("reduction_axis", logical_op->get_reduction_axes());
        break;
    }
    case OP_TYPEID::ArgMin:
    case OP_TYPEID::ArgMax:
    {
        const shared_ptr<op::util::IndexReduction> arg_op =
            static_pointer_cast<op::util::IndexReduction>(node);

        writer << print_table_row_value("reduction_axis", arg_op->get_reduction_axis())
               << table_row_begin() << font_small_begin
               << "idx_elem_type:" << arg_op->get_element_type() << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::LRN:
    {
        const shared_ptr<op::LRN> lrn_op = static_pointer_cast<op::LRN>(node);

        writer << print_table_row_value("nsize", lrn_op->get_nsize())
               << print_table_row_value("bias", lrn_op->get_bias())
               << print_table_row_value("alpha", lrn_op->get_alpha())
               << print_table_row_value("beta", lrn_op->get_beta());

        break;
    }
    case OP_TYPEID::OneHot:
    {
        const shared_ptr<op::OneHot> one_hot_op = static_pointer_cast<op::OneHot>(node);

        writer << print_table_row_value("one_hot_axis", one_hot_op->get_one_hot_axis());
        break;
    }
    case OP_TYPEID::Dot:
    {
        const shared_ptr<op::Dot> dot_op = static_pointer_cast<op::Dot>(node);

        writer << print_table_row_value("reduction_axes_count", dot_op->get_reduction_axes_count());
        break;
    }
    case OP_TYPEID::Constant:
    {
        size_t val_id = 0;
        const shared_ptr<op::Constant> constant_op = static_pointer_cast<op::Constant>(node);
        const vector<string>& values = constant_op->get_value_strings();

        // let's print no more than 3 items
        for (auto it = values.cbegin(); (it != values.cend()) && (val_id < 3); ++it)
        {
            writer << print_table_row_value("value[" + to_string(val_id) + "]", *it);
            ++val_id;
        }
        break;
    }
    case OP_TYPEID::Reshape:
    {
        const shared_ptr<op::Reshape> op_reshape = static_pointer_cast<op::Reshape>(node);

        writer << print_table_row_dims("broadcast_axes", op_reshape->get_input_order())
               << print_table_row_value("transpose", op_reshape->get_is_transpose());
        break;
    }
    case OP_TYPEID::Concat:
    {
        const shared_ptr<op::Concat> concat_op = static_pointer_cast<op::Concat>(node);

        writer << print_table_row_value("concat_axis", concat_op->get_concatenation_axis());
        break;
    }
    case OP_TYPEID::Reduce:
    {
        const shared_ptr<op::Reduce> red_op = static_pointer_cast<op::Reduce>(node);
        const AxisSet& axis = red_op->get_reduction_axes();

        writer << print_table_row_dims("reduction_axis", red_op->get_reduction_axes())
               << print_table_row_value("Function:TBD", 0);
        break;
    }
    case OP_TYPEID::ReduceWindow:
    {
        const shared_ptr<op::ReduceWindow> red_win_op = static_pointer_cast<op::ReduceWindow>(node);

        writer << print_table_row_dims("window_shape", red_win_op->get_window_shape())
               << print_table_row_dims("window_stride", red_win_op->get_window_movement_strides())
               << print_table_row_value("Function:TBD", 0);
        break;
    }
    case OP_TYPEID::Pad:
    {
        const shared_ptr<op::Pad> pad = static_pointer_cast<op::Pad>(node);

        writer << print_table_row_dims("pad_above", pad->get_padding_above())
               << print_table_row_dims("pad_below", pad->get_padding_below())
               << print_table_row_dims("pad_interior", pad->get_padding_interior());
        break;
    }
    case OP_TYPEID::Slice:
    {
        const shared_ptr<op::Slice> elem = static_pointer_cast<op::Slice>(node);

        writer << print_table_row_dims("upper_bounds", elem->get_upper_bounds())
               << print_table_row_dims("lower_bounds", elem->get_lower_bounds())
               << print_table_row_dims("strides", elem->get_strides());

        break;
    }
    case OP_TYPEID::Convolution:
    {
        const shared_ptr<op::Convolution> conv_op = static_pointer_cast<op::Convolution>(node);

        writer << print_table_row_dims("win_stride", conv_op->get_window_movement_strides())
               << print_table_row_dims("win_dilation", conv_op->get_window_dilation_strides())
               << print_table_row_dims("data_dilation", conv_op->get_data_dilation_strides())
               << print_table_row_dims("pad_above", conv_op->get_padding_above())
               << print_table_row_dims("pad_below", conv_op->get_padding_below());
        break;
    }
    case OP_TYPEID::ConvolutionBackpropFilters:
    {
        const shared_ptr<op::ConvolutionBackpropFilters> conv_op_filt =
            static_pointer_cast<op::ConvolutionBackpropFilters>(node);

        writer << print_table_row_dims("filters_shape", conv_op_filt->get_filters_shape())
               << print_table_row_dims("window_movement_strides_forward",
                                       conv_op_filt->get_window_movement_strides_forward())
               << print_table_row_dims("window_dilation_strides_forward",
                                       conv_op_filt->get_window_dilation_strides_forward())
               << print_table_row_dims("data_dilation_strides_forward",
                                       conv_op_filt->get_data_dilation_strides_forward())
               << print_table_row_dims("pad_above_forward",
                                       conv_op_filt->get_padding_above_forward())
               << print_table_row_dims("pad_below_forward",
                                       conv_op_filt->get_padding_below_forward())
               << print_table_row_dims("window_movement_strides_backward",
                                       conv_op_filt->get_window_movement_strides_backward())
               << print_table_row_dims("window_dilation_strides_backward",
                                       conv_op_filt->get_window_dilation_strides_backward())
               << print_table_row_dims("data_dilation_strides_backward",
                                       conv_op_filt->get_data_dilation_strides_backward())
               << print_table_row_dims("padding_above_backward",
                                       conv_op_filt->get_padding_above_backward())
               << print_table_row_dims("padding_below_backward",
                                       conv_op_filt->get_padding_below_backward());
        break;
    }
    case OP_TYPEID::ConvolutionBackpropData:
    {
        const shared_ptr<op::ConvolutionBackpropData> conv_op_data =
            static_pointer_cast<op::ConvolutionBackpropData>(node);

        writer << print_table_row_dims("data_batch_shape", conv_op_data->get_data_batch_shape())
               << print_table_row_dims("window_movement_strides_forward",
                                       conv_op_data->get_window_movement_strides_forward())
               << print_table_row_dims("window_dilation_strides_forward",
                                       conv_op_data->get_window_dilation_strides_forward())
               << print_table_row_dims("data_dilation_strides_forward",
                                       conv_op_data->get_data_dilation_strides_forward())
               << print_table_row_dims("pad_above_forward",
                                       conv_op_data->get_padding_above_forward())
               << print_table_row_dims("pad_below_forward",
                                       conv_op_data->get_padding_below_forward())
               << print_table_row_dims("window_movement_strides_backward",
                                       conv_op_data->get_window_movement_strides_backward())
               << print_table_row_dims("window_dilation_strides_backward",
                                       conv_op_data->get_window_dilation_strides_backward())
               << print_table_row_dims("data_dilation_strides_backward",
                                       conv_op_data->get_data_dilation_strides_backward())
               << print_table_row_dims("padding_above_backward",
                                       conv_op_data->get_padding_above_backward())
               << print_table_row_dims("padding_below_backward",
                                       conv_op_data->get_padding_below_backward());
        break;
    }
    case OP_TYPEID::UNDEFINED_OP:
    default:
    {
        ; // Some operations are not defined in ngraph/op/op_tbl.hpp
    }
    }
}

void print_node(ostringstream& writer, const shared_ptr<Node>& node)
{
    writer << node->get_name() << " [";

    if (node->is_parameter())
    {
        writer << "shape=box color=blue ";
    }
    else if (node->is_output())
    {
        writer << "shape=box style=filled fillcolor=pink ";
    }
    else
    {
        writer << "shape=ellipse color=black";
    }

    // Print text inside figure using HTML layout
    writer << " label=<" << table_begin;

    if (!node->get_inputs().empty())
    {
        size_t arg_idx = 0;
        for (const descriptor::Input& op_input : node->get_inputs())
        {
            writer << table_row_begin() << font_small_begin
                   << op_input.get_element_type().c_type_string() << " input" << arg_idx
                   << vector_to_string(op_input.get_shape()) << font_end << table_row_end;
            ++arg_idx;
        }
    }

    if (!node->get_outputs().empty())
    {
        size_t arg_idx = 0;
        for (const descriptor::Output& op_output : node->get_outputs())
        {
            writer << table_row_begin() << font_small_begin
                   << op_output.get_element_type().c_type_string() << " output" << arg_idx
                   << vector_to_string(op_output.get_shape()) << font_end << table_row_end;
            ++arg_idx;
        }
    }

    writer << table_row_begin("center") << node->get_name() << table_row_end;

    print_node_parameters(writer, node);

    writer << table_end;
    writer << " >];\n";
}

void runtime::intelgpu::visualize_tree(const shared_ptr<Function>& func,
                                       const string& file_prefix,
                                       const string& file_suffix)
{
    map<string, size_t> operations;
    ostringstream writer;

    // Begin of the main graph
    writer << "digraph ngraph\n{\nsplines=\"line\";\n\n";

    for (const shared_ptr<Node> op : func->get_ordered_ops())
    {
        print_node(writer, op);

        for (const descriptor::Input& op_input : op->get_inputs())
        {
            writer << op_input.get_output().get_node()->get_name() << " -> " << op->get_name()
                   << ";\n";
        }

        // collect summary statistic for operations used in the graph
        const string op_name = op->get_name().substr(0, op->get_name().find_first_of("_"));
        auto it = operations.find(op_name);
        if (it == operations.end())
        {
            it = operations.emplace(op_name, 0).first;
        }
        ++(it->second);
    }

    // print summary with operations used
    writer << "\nsubgraph clusterFooter\n{\nmargin=0\nstyle=\"invis\"\nLEGEND ["
           << "shape=box style=filled fillcolor=gray margin=0 label=<" << table_begin
           << table_row_begin("center") << "Operations summary" << table_row_end;
    size_t total_op_count = 0;
    for (const auto& it : operations)
    {
        writer << table_row_begin() << font_small_begin << it.first << ":" << font_end << cell_end
               << cell_begin() << font_small_begin << it.second << font_end << table_row_end;

        total_op_count += it.second;
    }
    writer << table_row_begin() << "Total:" << cell_end << cell_begin() << total_op_count
           << table_row_end << table_end << "\n>];\n}\n";

    // End of the main graph
    writer << "}\n";

    ofstream out_file(file_prefix + func->get_name() + file_suffix + ".dot");
    if (out_file)
    {
        out_file << writer.str();
        out_file.close();
    }
}
