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
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;
using namespace std;

#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/op_tbl.hpp"
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
        throw unsupported_op("Unsupported op '" + s + "'");
    }
    return it->second;
}

static const string table_begin = "<table border=\"0\">";
static const string table_end = "</table>";
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
    return string("<tr>") + cell_begin(align);
}

void print_node_parameters(ostringstream& writer, const shared_ptr<Node>& node)
{
    switch (get_typeid(node->description()))
    {
    case OP_TYPEID::BatchNormTrainingBackprop:
    {
        const shared_ptr<op::BatchNormTrainingBackprop> batch_norm =
            static_pointer_cast<op::BatchNormTrainingBackprop>(node);
        const double eps = batch_norm->get_eps_value();

        writer << table_row_begin() << font_small_begin << "EPS:" << eps << font_end
               << table_row_end;
        break;
    }
    case OP_TYPEID::BatchNormInference:
    case OP_TYPEID::BatchNormTraining:
    {
        const shared_ptr<op::BatchNormBase> batch_norm =
            static_pointer_cast<op::BatchNormBase>(node);
        const double eps = batch_norm->get_eps_value();

        writer << table_row_begin() << font_small_begin << "EPS:" << eps << font_end
               << table_row_end;
        break;
    }
    case OP_TYPEID::GetOutputElement:
    {
        const shared_ptr<op::GetOutputElement> elem =
            static_pointer_cast<op::GetOutputElement>(node);

        writer << table_row_begin() << font_small_begin << "element:" << elem->get_n() << font_end
               << table_row_end;
        break;
    }
    case OP_TYPEID::MaxPool:
    {
        const shared_ptr<op::MaxPool> max_pool = static_pointer_cast<op::MaxPool>(node);

        writer << table_row_begin() << font_small_begin << "win_shape"
               << vector_to_string(max_pool->get_window_shape()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "win_strides"
               << vector_to_string(max_pool->get_window_movement_strides()) << font_end
               << table_row_end << table_row_begin() << font_small_begin << "pad_above"
               << vector_to_string(max_pool->get_padding_above()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "pad_below"
               << vector_to_string(max_pool->get_padding_below()) << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::MaxPoolBackprop:
    {
        const shared_ptr<op::MaxPoolBackprop> max_pool_b =
            static_pointer_cast<op::MaxPoolBackprop>(node);

        writer << table_row_begin() << font_small_begin << "win_shape"
               << vector_to_string(max_pool_b->get_window_shape()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "win_strides"
               << vector_to_string(max_pool_b->get_window_movement_strides()) << font_end
               << table_row_end << table_row_begin() << font_small_begin << "pad_above"
               << vector_to_string(max_pool_b->get_padding_above()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "pad_below"
               << vector_to_string(max_pool_b->get_padding_below()) << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::AvgPool:
    {
        const shared_ptr<op::AvgPool> avg_pool = static_pointer_cast<op::AvgPool>(node);

        writer << table_row_begin() << font_small_begin << "win_shape"
               << vector_to_string(avg_pool->get_window_shape()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "win_strides"
               << vector_to_string(avg_pool->get_window_movement_strides()) << font_end
               << table_row_end << table_row_begin() << font_small_begin << "pad_above"
               << vector_to_string(avg_pool->get_padding_above()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "pad_below"
               << vector_to_string(avg_pool->get_padding_below()) << font_end << table_row_end
               << table_row_begin() << font_small_begin
               << "pad_included:" << avg_pool->get_include_padding_in_avg_computation() << font_end
               << table_row_end;
        break;
    }
    case OP_TYPEID::AvgPoolBackprop:
    {
        const shared_ptr<op::AvgPoolBackprop> avg_pool_b =
            static_pointer_cast<op::AvgPoolBackprop>(node);

        writer << table_row_begin() << font_small_begin << "win_shape"
               << vector_to_string(avg_pool_b->get_window_shape()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "win_strides"
               << vector_to_string(avg_pool_b->get_window_movement_strides()) << font_end
               << table_row_end << table_row_begin() << font_small_begin << "pad_above"
               << vector_to_string(avg_pool_b->get_padding_above()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "pad_below"
               << vector_to_string(avg_pool_b->get_padding_below()) << font_end << table_row_end
               << table_row_begin() << font_small_begin
               << "pad_included:" << avg_pool_b->get_include_padding_in_avg_computation()
               << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::Broadcast:
    {
        const shared_ptr<op::Broadcast> broadcast = static_pointer_cast<op::Broadcast>(node);

        writer << table_row_begin() << font_small_begin << "broadcast_axis"
               << vector_to_string(broadcast->get_broadcast_axes()) << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::Sum:
    {
        const shared_ptr<op::Sum> sum = static_pointer_cast<op::Sum>(node);

        writer << table_row_begin() << font_small_begin << "reduction_axis"
               << vector_to_string(sum->get_reduction_axes()) << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::Product:
    {
        const shared_ptr<op::Product> prod = static_pointer_cast<op::Product>(node);

        writer << table_row_begin() << font_small_begin << "reduction_axis"
               << vector_to_string(prod->get_reduction_axes()) << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::Reshape:
    {
        const shared_ptr<op::Reshape> op_reshape = static_pointer_cast<op::Reshape>(node);

        writer << table_row_begin() << font_small_begin
               << "broadcast_axes:" << vector_to_string(op_reshape->get_input_order()) << font_end
               << table_row_end << table_row_begin() << font_small_begin
               << "transpose:" << op_reshape->get_is_transpose() << font_end << table_row_end;
        break;
    }
    case OP_TYPEID::Concat:
    {
        const shared_ptr<op::Concat> concat_op = static_pointer_cast<op::Concat>(node);

        writer << table_row_begin() << font_small_begin
               << "concat_axis:" << concat_op->get_concatenation_axis() << font_end
               << table_row_end;
        break;
    }
    case OP_TYPEID::Convolution:
    {
        const shared_ptr<op::Convolution> conv_op = static_pointer_cast<op::Convolution>(node);

        writer << table_row_begin() << font_small_begin << "win_stride"
               << vector_to_string(conv_op->get_window_movement_strides()) << font_end
               << table_row_end << table_row_begin() << font_small_begin << "win_dilation"
               << vector_to_string(conv_op->get_window_dilation_strides()) << font_end
               << table_row_end << table_row_begin() << font_small_begin << "data_dilation"
               << vector_to_string(conv_op->get_data_dilation_strides()) << font_end
               << table_row_end << table_row_begin() << font_small_begin << "pad_below"
               << vector_to_string(conv_op->get_padding_below()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "pad_above"
               << vector_to_string(conv_op->get_padding_above()) << font_end << table_row_end
               << table_row_begin() << font_small_begin << "def_val"
               << vector_to_string(conv_op->get_default_value()->get_shape()) << font_end
               << table_row_end;
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
    writer << " >]\n";
}

void runtime::intelgpu::visualize_tree(const shared_ptr<Function>& func,
                                       const string& file_prefix,
                                       const string& file_suffix)
{
    map<string, size_t> operations;
    ostringstream writer;

    // Begin of the main graph
    writer << "digraph ngraph\n{\n";

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
    writer << "subgraph clusterFooter\n{\nmargin=0\nstyle=\"invis\"\nLEGEND ["
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
           << table_row_end << table_end << " >]\n}\n";

    // End of the main graph
    writer << "}\n";

    ofstream out_file(file_prefix + func->get_name() + file_suffix + ".dot");
    if (out_file)
    {
        out_file << writer.str();
        out_file.close();
    }
}
