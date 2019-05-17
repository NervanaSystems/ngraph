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

#include <iomanip>

#include <CPP/activation.hpp>
#include <CPP/activation_grad.hpp>
#include <CPP/arg_max_min.hpp>
#include <CPP/batch_norm.hpp>
#include <CPP/border.hpp>
#include <CPP/broadcast.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/convolution.hpp>
#include <CPP/convolution_grad_input.hpp>
#include <CPP/convolution_grad_weights.hpp>
#include <CPP/crop.hpp>
#include <CPP/data.hpp>
#include <CPP/eltwise.hpp>
#include <CPP/gemm.hpp>
#include <CPP/input_layout.hpp>
#include <CPP/layout.hpp>
#include <CPP/lrn.hpp>
#include <CPP/mutable_data.hpp>
#include <CPP/permute.hpp>
#include <CPP/pooling.hpp>
#include <CPP/reshape.hpp>
#include <CPP/select.hpp>
#include <CPP/softmax.hpp>
#include <CPP/topology.hpp>

#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/batch_fusion.hpp"
#include "ngraph/pass/core_fusion.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/fused_op_decomposition.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/runtime/backend_manager.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_executable.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"
#include "ngraph/runtime/intelgpu/visualize_tree.hpp"

#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/all.hpp"
#include "ngraph/op/and.hpp"
#include "ngraph/op/any.hpp"
#include "ngraph/op/argmax.hpp"
#include "ngraph/op/argmin.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dequantize.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/equal.hpp"
#include "ngraph/op/erf.hpp"
#include "ngraph/op/fused/conv_fused.hpp"
#include "ngraph/op/fused/depth_to_space.hpp"
#include "ngraph/op/fused/elu.hpp"
#include "ngraph/op/fused/gemm.hpp"
#include "ngraph/op/fused/grn.hpp"
#include "ngraph/op/fused/group_conv.hpp"
#include "ngraph/op/fused/hard_sigmoid.hpp"
#include "ngraph/op/fused/mvn.hpp"
#include "ngraph/op/fused/normalize.hpp"
#include "ngraph/op/fused/scale_shift.hpp"
#include "ngraph/op/fused/space_to_depth.hpp"
#include "ngraph/op/fused/squeeze.hpp"
#include "ngraph/op/fused/unsqueeze.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/greater.hpp"
#include "ngraph/op/greater_eq.hpp"
#include "ngraph/op/less.hpp"
#include "ngraph/op/less_eq.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/not_equal.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/or.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/quantize.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/reverse_sequence.hpp"
#include "ngraph/op/select.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/op/topk.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using intelgpu_space = runtime::intelgpu::IntelGPULayout;

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
#define NGRAPH_OP(a, b) a,
enum class OP_TYPEID
{
#include "ngraph/op/fused_op_tbl.hpp"
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
#include "ngraph/op/fused_op_tbl.hpp"
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

static void do_eltwise_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 const string& custom_op,
                                 bool function_operation,
                                 cldnn::eltwise_mode mode)
{
    runtime::intelgpu::arguments_check(op, 2, 1);

    if (op->get_input_element_type(0) != element::f32 ||
        op->get_input_element_type(1) != element::f32 ||
        op->get_output_element_type(0) != element::f32)
    {
        runtime::intelgpu::do_eltwise_kernel(topology,
                                             op->get_input_tensor_name(0),
                                             op->get_input_shape(0),
                                             op->get_input_element_type(0),
                                             op->get_input_tensor_name(1),
                                             op->get_input_shape(1),
                                             op->get_output_tensor_name(0),
                                             op->get_output_shape(0),
                                             op->get_output_element_type(0),
                                             custom_op,
                                             function_operation);
    }
    else
    {
        const cldnn::eltwise op_eltwise(
            op->get_output_tensor_name(0),
            {op->get_input_tensor_name(0), op->get_input_tensor_name(1)},
            mode);
        topology.add(op_eltwise);
    }
}

static void do_cldnn_unary(cldnn::topology& topology,
                           const shared_ptr<Node>& op,
                           cldnn_activation_func mode,
                           const cldnn_activation_additional_params& param = {0.f, 0.f})
{
    runtime::intelgpu::arguments_check(op, 1, 1);

    const cldnn::activation cldnn_unary(
        op->get_output_tensor_name(0), op->get_input_tensor_name(0), mode, param);
    topology.add(cldnn_unary);
}

static bool has_non_zero(const Shape& shape)
{
    return accumulate(shape.begin(), shape.end(), 0);
}

static void
    do_custom_unary(cldnn::topology& topology, const shared_ptr<Node>& op, const string& operation)
{
    runtime::intelgpu::arguments_check(op, 1, 1);

    runtime::intelgpu::do_custom_unary_operation(topology,
                                                 op->get_input_tensor_name(0),
                                                 op->get_input_shape(0),
                                                 op->get_input_element_type(0),
                                                 op->get_output_tensor_name(0),
                                                 op->get_output_shape(0),
                                                 op->get_output_element_type(0),
                                                 operation);
}

static void do_universal_unary(cldnn::topology& topology,
                               const shared_ptr<Node>& op,
                               const string& operation,
                               cldnn_activation_func mode,
                               bool force_custom = false,
                               const cldnn_activation_additional_params& param = {0.f, 0.f})
{
    runtime::intelgpu::arguments_check(op, 1, 1);

    if (force_custom || (op->get_input_element_type(0) != element::f32))
    {
        do_custom_unary(topology, op, operation);
    }
    else
    {
        do_cldnn_unary(topology, op, mode, param);
    }
}

static void do_pooling_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 const Shape& pool_shape,
                                 const Strides& pool_strides,
                                 const Shape& pad_below,
                                 const cldnn::pooling_mode mode)
{
    runtime::intelgpu::arguments_check(op, 1, 1);

    const cldnn::tensor output_size = intelgpu_space::create_cldnn_tensor(op->get_output_shape(0));
    const cldnn::tensor input_offset = intelgpu_space::create_cldnn_offset(pad_below);
    const cldnn::tensor size = intelgpu_space::create_cldnn_tensor(pool_shape);
    const cldnn::tensor stride = intelgpu_space::create_cldnn_tensor(pool_strides);

    const cldnn::pooling cldnn_pooling(op->get_output_tensor_name(0),
                                       op->get_input_tensor_name(0),
                                       mode,
                                       size,
                                       stride,
                                       input_offset,
                                       output_size);
    topology.add(cldnn_pooling);
}

template <typename OP>
static void do_logical_operation(runtime::intelgpu::CustomKernels& kern, const shared_ptr<Node>& op)
{
    runtime::intelgpu::arguments_check(op, 2, 1);

    kern.emit<OP>(static_pointer_cast<OP>(op));
}

// This function needed to only change the name of the data in topology
// No real data copy needed
static void do_equal_propagation(cldnn::topology& topology,
                                 const string& input_name,
                                 const string& output_name)
{
    const vector<cldnn::primitive_id> input_names(1, input_name);

    const cldnn::concatenation op_concat(output_name, input_names, cldnn::concatenation::along_x);
    topology.add(op_concat);
}

extern "C" runtime::BackendConstructor* get_backend_constructor_pointer()
{
    class IntelGPUBackendConstructor : public runtime::BackendConstructor
    {
    public:
        std::shared_ptr<runtime::Backend> create(const std::string& config) override
        {
            return std::make_shared<runtime::intelgpu::IntelGPUBackend>();
        }
    };

    static unique_ptr<runtime::BackendConstructor> s_backend_constructor(
        new IntelGPUBackendConstructor());
    return s_backend_constructor.get();
}

runtime::intelgpu::IntelGPUBackend::IntelGPUBackend()
{
    bool profiling = false;

    // This should be used to allow nbench work with "--timing_detail" option
    if (getenv("NGRAPH_INTELGPU_STAT") != nullptr)
    {
        profiling = true;
    }

    // Print out default profile and statistic to the output
    if (getenv("NGRAPH_INTELGPU_PROFILE") != nullptr)
    {
        profiling = true;
        m_profile_enable = true;
    }

    // Control the number of lines in ::call profile
    const char* profile_lines_count = getenv("NGRAPH_INTELGPU_PROFILE_LINES");
    if (profile_lines_count != nullptr)
    {
        profiling = true;
        m_profile_enable = true;
        m_profile_lines_limit_count = strtol(profile_lines_count, nullptr, 10);
    }

    // Disables the backend Function (graph) level optimizations
    // 0 or undefined - All optimization passes are enabled
    // 1 - Disable optimization passes except FusedOpDecomposition
    // >1 - Disable all optimization passes
    const char* disable_backend_optimizations = getenv("NGRAPH_INTELGPU_DISABLE_OPTIMIZATIONS");
    if (disable_backend_optimizations != nullptr)
    {
        m_disable_backend_optimizations = strtol(disable_backend_optimizations, nullptr, 10);
    }

    // Disables clDNN (cldnn::network) level optimizations
    if (getenv("NGRAPH_INTELGPU_CLDNN_DISABLE_OPTIMIZATIONS") != nullptr)
    {
        m_cldnn_graph_optimize = false;
    }

    // Dumps the input Function into Graphviz format
    if (getenv("NGRAPH_INTELGPU_DUMP_FUNCTION") != nullptr)
    {
        m_dump_graph_enable = true;
    }

    // Dumps the clDNN internal logs into directory
    if (getenv("NGRAPH_INTELGPU_CLDNN_DUMP") != nullptr)
    {
        file_util::make_directory(m_cldnn_dump_dir);
        m_cldnn_dump_enable = true;
    }

    // Delete compiled Function from the cache after execution.
    // It helps in cases where a lot of small functions used
    // in case of memory consumption. It slow overall execution
    // because Function compilation required every time
    if (getenv("NGRAPH_INTELGPU_FUNCTION_CACHE_DISABLE") != nullptr)
    {
        m_function_cache_disabled = true;
    }

    cldnn::engine_configuration cldnn_configuration(profiling,
                                                    false,
                                                    m_cldnn_dump_enable,
                                                    string(),
                                                    string(),
                                                    true,
                                                    string(),
                                                    m_cldnn_dump_dir);
    cldnn_engine = make_shared<cldnn::engine>(cldnn_configuration);
}

shared_ptr<runtime::Tensor>
    runtime::intelgpu::IntelGPUBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *cldnn_engine, nullptr);
}

shared_ptr<runtime::Tensor> runtime::intelgpu::IntelGPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *cldnn_engine, memory_pointer);
}

shared_ptr<runtime::Executable>
    runtime::intelgpu::IntelGPUBackend::compile(shared_ptr<Function> func, bool enable_timing)
{
    shared_ptr<runtime::Executable> rc;

    auto it = cldnn_networks.find(func);
    if (it != cldnn_networks.end())
    {
        return it->second;
    }

    set<cldnn::primitive_id> func_output_names;
    cldnn::topology topology;
    CustomKernels kern(topology);
    stopwatch timer_compile;
    double consumed_memory = 0.0;
    double compilation_time = 0.0;

    if (m_profile_enable)
    {
        consumed_memory = runtime::intelgpu::get_max_memory_rss();
        timer_compile.start();
    }

    if (m_dump_graph_enable)
    {
        visualize_tree(func, "intelgpu_", "_orig");
    }

    ngraph::pass::Manager pass_manager;

    if (m_disable_backend_optimizations < 2)
    {
        pass_manager.register_pass<ngraph::pass::FusedOpDecomposition>(
            IntelGPUBackend::is_supported_impl);
    }

    if (m_disable_backend_optimizations < 1)
    {
        pass_manager.register_pass<ngraph::pass::NopElimination>();
        pass_manager.register_pass<ngraph::pass::BatchFusion>();
        pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
        pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
        pass_manager.register_pass<ngraph::pass::ReshapeElimination>();
        pass_manager.register_pass<ngraph::pass::CoreFusion>(ngraph::pass::ALL_FUSIONS);

        // GetOutputElementElimination must be after CommonSubexpressionElimination
        pass_manager.register_pass<ngraph::pass::GetOutputElementElimination>();
    }

    if (m_disable_backend_optimizations < 2)
    {
        pass_manager.run_passes(func);

        if (m_dump_graph_enable)
        {
            visualize_tree(func, "intelgpu_", "_opt");
        }
    }

    for (shared_ptr<Node> op : func->get_ops())
    {
        const OP_TYPEID op_type_id = get_typeid(op->description());
// We want to check that every OP_TYPEID enumeration is included in the list.
// These GCC flags enable compile-time checking so that if an enumeration
// is not in the list an error is generated.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
        switch (op_type_id)
        {
        case OP_TYPEID::Parameter:
        {
            arguments_check(op, 0, 1);

            const string& element_name = op->get_output_tensor_ptr()->get_name();
            const cldnn::layout element_layout =
                IntelGPULayout::create_cldnn_layout(op->get_element_type(), op->get_shape());

            const cldnn::input_layout op_layout(element_name, element_layout);
            topology.add(op_layout);
            break;
        }
        case OP_TYPEID::Result:
        {
            arguments_check(op, 1, 1);

            func_output_names.insert(op->get_input_tensor_name(0));
            break;
        }
        case OP_TYPEID::GetOutputElement:
        {
            if (op->get_inputs().empty() || op->get_outputs().size() != 1)
            {
                arguments_check(op, 1, 1); // at least one input and exact one output expected
            }

            const shared_ptr<op::GetOutputElement> elem =
                static_pointer_cast<op::GetOutputElement>(op);

            do_equal_propagation(
                topology, op->get_input_tensor_name(elem->get_n()), op->get_output_tensor_name(0));
            break;
        }
        case OP_TYPEID::Slice:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Slice> elem = static_pointer_cast<op::Slice>(op);
            const Coordinate& lower_bounds = elem->get_lower_bounds();
            const Coordinate& upper_bounds = elem->get_upper_bounds();
            const Strides& strides = elem->get_strides();

            if (op->get_input_shape(0).empty() || op->get_output_shape(0).empty() ||
                lower_bounds.empty() || upper_bounds.empty() || strides.empty())
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else
            {
                kern.emit<op::Slice>(elem);
            }
            break;
        }
        case OP_TYPEID::Select:
        {
            arguments_check(op, 3, 1);

            if (op->get_output_element_type(0) != element::f32)
            {
                kern.emit<op::Select>(static_pointer_cast<op::Select>(op));
            }
            else
            {
                const cldnn::select cldnn_select(op->get_output_tensor_name(0),
                                                 op->get_input_tensor_name(1),
                                                 op->get_input_tensor_name(2),
                                                 op->get_input_tensor_name(0));
                topology.add(cldnn_select);
            }
            break;
        }
        case OP_TYPEID::Reverse:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Reverse> reverse_op = static_pointer_cast<op::Reverse>(op);
            const AxisSet& reversed_axes = reverse_op->get_reversed_axes();

            if (reversed_axes.empty())
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else
            {
                do_reverse_operation(topology,
                                     op->get_input_tensor_name(0),
                                     op->get_input_shape(0),
                                     op->get_input_element_type(0),
                                     op->get_output_tensor_name(0),
                                     op->get_output_shape(0),
                                     op->get_output_element_type(0),
                                     reversed_axes);
            }
            break;
        }
        case OP_TYPEID::ReverseSequence:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::ReverseSequence> revseq_op =
                static_pointer_cast<op::ReverseSequence>(op);
            const size_t batch_axis = revseq_op->get_batch_axis();
            const size_t seq_axis = revseq_op->get_sequence_axis();

            do_reverse_sequence_operation(topology,
                                          op->get_input_tensor_name(0),
                                          op->get_input_shape(0),
                                          op->get_input_element_type(0),
                                          op->get_input_tensor_name(1),
                                          op->get_input_shape(1),
                                          op->get_input_element_type(1),
                                          op->get_output_tensor_name(0),
                                          op->get_output_shape(0),
                                          op->get_output_element_type(0),
                                          seq_axis,
                                          batch_axis);
            break;
        }
        case OP_TYPEID::Convert:
        {
            arguments_check(op, 1, 1);

            if (op->get_input_element_type(0) == op->get_output_element_type(0))
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else
            {
                do_convert_operation(topology,
                                     op->get_input_tensor_name(0),
                                     op->get_input_shape(0),
                                     op->get_input_element_type(0),
                                     op->get_output_tensor_name(0),
                                     op->get_output_shape(0),
                                     op->get_output_element_type(0));
            }
            break;
        }
        case OP_TYPEID::Concat:
        {
            if (op->get_inputs().empty() || op->get_outputs().size() != 1)
            {
                arguments_check(op, 1, 1);
            }

            const shared_ptr<op::Concat> concat_op = static_pointer_cast<op::Concat>(op);
            const size_t ngraph_concat_axis = concat_op->get_concatenation_axis();

            if (!shape_size(op->get_output_shape(0)) ||
                (op->get_input_element_type(0) != element::f32) ||
                op->get_output_shape(0).size() > 4)
            {
                vector<string> input_names;
                vector<Shape> input_shapes;

                for (auto const& input : op->get_inputs())
                {
                    const Shape& input_shape = input.get_tensor().get_shape();
                    if (shape_size(input_shape))
                    {
                        input_names.push_back(input.get_tensor().get_name());
                        input_shapes.push_back(input_shape);
                    }
                }

                if (input_names.empty())
                {
                    do_equal_propagation(
                        topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
                }
                else
                {
                    do_concat_operation(topology,
                                        input_names,
                                        input_shapes,
                                        op->get_output_tensor_name(0),
                                        op->get_output_shape(0),
                                        op->get_output_element_type(0),
                                        ngraph_concat_axis);
                }
            }
            else
            {
                // All input shapes must be the same
                // if shape is empty (means Shape{}) in this case treat its size as 1
                const size_t ngraph_tensor_dims =
                    op->get_input_shape(0).empty() ? 1 : op->get_input_shape(0).size();
                vector<cldnn::primitive_id> inputs;

                cldnn::concatenation::concatenation_axis cldnn_axis =
                    intelgpu_space::get_cldnn_axis(ngraph_tensor_dims, ngraph_concat_axis);

                for (auto const& input : op->get_inputs())
                {
                    const Shape& input_shape = input.get_shape();
                    if (shape_size(input_shape))
                    {
                        inputs.push_back(input.get_tensor().get_name());
                    }
                }

                if (inputs.empty())
                {
                    do_equal_propagation(
                        topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
                }
                else
                {
                    const cldnn::concatenation cldnn_concat(
                        op->get_output_tensor_name(0), inputs, cldnn_axis);
                    topology.add(cldnn_concat);
                }
            }
            break;
        }
        case OP_TYPEID::Softmax:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Softmax> softmax_op = static_pointer_cast<op::Softmax>(op);
            const AxisSet& axes = softmax_op->get_axes();
            const size_t axes_size = axes.size();
            const size_t shape_dim_count = op->get_input_shape(0).size();

            // clDNN has limited support for Softmax operation
            // following are the checks to go with custom kernel
            if ((shape_dim_count > 3) || ((shape_dim_count == 3) && (axes_size == 2)) ||
                (op->get_input_element_type(0) != element::f32))
            {
                kern.emit<op::Softmax>(softmax_op);
            }
            else
            {
                cldnn::softmax::dimension_t dimension = cldnn::softmax::normalize_fyx;
                if (axes_size == 1)
                {
                    size_t axes_idx = shape_dim_count - *(axes.begin()) - 1;
                    switch (axes_idx)
                    {
                    case 0: dimension = cldnn::softmax::normalize_x; break;
                    case 1: dimension = cldnn::softmax::normalize_y; break;
                    case 2: dimension = cldnn::softmax::normalize_f; break;
                    default:
                        throw invalid_argument("Softmax operation: wrong axis " +
                                               to_string(axes_idx));
                    }
                }

                const cldnn::softmax cldnn_softmax(
                    op->get_output_tensor_name(0), op->get_input_tensor_name(0), dimension);
                topology.add(cldnn_softmax);
            }
            break;
        }
        case OP_TYPEID::Add:
        {
            do_eltwise_operation(topology, op, "+", false, cldnn::eltwise_mode::sum);
            break;
        }
        case OP_TYPEID::Subtract:
        {
            do_eltwise_operation(topology, op, "-", false, cldnn::eltwise_mode::sub);
            break;
        }
        case OP_TYPEID::Multiply:
        {
            do_eltwise_operation(topology, op, "*", false, cldnn::eltwise_mode::prod);
            break;
        }
        case OP_TYPEID::Divide:
        {
            do_eltwise_operation(topology, op, "/", false, cldnn::eltwise_mode::div);
            break;
        }
        case OP_TYPEID::Maximum:
        {
            do_eltwise_operation(topology, op, "max", true, cldnn::eltwise_mode::max);
            break;
        }
        case OP_TYPEID::Minimum:
        {
            do_eltwise_operation(topology, op, "min", true, cldnn::eltwise_mode::min);
            break;
        }
        case OP_TYPEID::Power:
        {
            do_eltwise_operation(topology, op, "pow", true, cldnn::eltwise_mode::pow);
            break;
        }
        case OP_TYPEID::Constant:
        {
            arguments_check(op, 0, 1);

            const shared_ptr<op::Constant> constant_inst = static_pointer_cast<op::Constant>(op);
            void* memory_pointer = const_cast<void*>(constant_inst->get_data_ptr());

            const cldnn::layout layout = IntelGPULayout::create_cldnn_layout(
                op->get_output_element_type(0), op->get_output_shape(0));
            const cldnn::memory mem(
                cldnn::memory::attach<void>(layout, memory_pointer, layout.bytes_count()));

            const cldnn::data op_const(op->get_output_tensor_name(0), mem);
            topology.add(op_const);
            break;
        }
        case OP_TYPEID::Dot:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::Dot> dot_inst = static_pointer_cast<op::Dot>(op);
            const size_t axes_count = dot_inst->get_reduction_axes_count();
            const Shape& input0_shape = op->get_input_shape(0);
            const Shape& input1_shape = op->get_input_shape(1);
            const size_t input0_elem_count = shape_size(input0_shape);
            const size_t input1_elem_count = shape_size(input1_shape);

            if (op->get_input_element_type(0) == element::f32 &&
                op->get_input_element_type(1) == element::f32 &&
                op->get_output_element_type(0) == element::f32 && input0_elem_count &&
                input1_elem_count && (axes_count < 2) && (input0_shape.size() < 3) &&
                (input1_shape.size() < 3))
            {
                bool transpose0 = false;
                bool transpose1 = false;

                // If we have A[5] and B[] here, in cldnn we have A[1, 1, 1, 5] and B[1, 1, 1, 1]
                // it needs to be reshaped into A[1, 1, 5, 1] and B[1, 1, 1, 1]
                if ((input0_shape.size() == 1) && input1_shape.empty())
                {
                    transpose0 = true;
                }

                // If we have A[5] and B[5] here, in cldnn we have A[1, 1, 1, 5] and B[1, 1, 1, 5]
                // it needs to be reshaped into A[1, 1, 1, 5] and B[1, 1, 5, 1]
                if (!input0_shape.empty() && (input1_shape.size() == 1))
                {
                    transpose1 = true;
                }

                const cldnn::gemm dot_op(op->get_output_tensor_name(0),
                                         op->get_input_tensor_name(0),
                                         op->get_input_tensor_name(1),
                                         transpose0,
                                         transpose1);
                topology.add(dot_op);
            }
            else
            {
                do_dot_operation(topology,
                                 op->get_input_tensor_name(0),
                                 op->get_input_shape(0),
                                 op->get_input_tensor_name(1),
                                 op->get_input_shape(1),
                                 op->get_output_tensor_name(0),
                                 op->get_output_shape(0),
                                 op->get_output_element_type(0),
                                 axes_count);
            }
            break;
        }
        case OP_TYPEID::Gemm:
        {
            arguments_check(op, 3, 1);

            const shared_ptr<op::Gemm> gemm_op = static_pointer_cast<op::Gemm>(op);
            const double alpha = gemm_op->get_alpha();
            const double beta = gemm_op->get_beta();
            const bool transA = gemm_op->get_transA();
            const bool transB = gemm_op->get_transB();

            if (op->get_input_element_type(0) == element::f32 &&
                op->get_input_element_type(1) == element::f32 &&
                op->get_input_element_type(2) == element::f32 &&
                op->get_output_element_type(0) == element::f32)
            {
                const cldnn::gemm gemm_op(op->get_output_tensor_name(0),
                                          op->get_input_tensor_name(0),
                                          op->get_input_tensor_name(1),
                                          op->get_input_tensor_name(2),
                                          transA,
                                          transB,
                                          (float)alpha,
                                          (float)beta);
                topology.add(gemm_op);
            }
            else
            {
                if (alpha == 1.0 && beta == 0.0 && transA == false && transB == false)
                {
                    do_dot_operation(topology,
                                     op->get_input_tensor_name(0),
                                     op->get_input_shape(0),
                                     op->get_input_tensor_name(1),
                                     op->get_input_shape(1),
                                     op->get_output_tensor_name(0),
                                     op->get_output_shape(0),
                                     op->get_output_element_type(0),
                                     0);
                }
                else
                {
                    kern.emit<op::Gemm>(gemm_op);
                }
            }

            break;
        }
        case OP_TYPEID::MaxPool:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::MaxPool> max_pool = static_pointer_cast<op::MaxPool>(op);

            if ((op->get_input_shape(0).size() > 4) ||
                (op->get_output_element_type(0) != element::f32))
            {
                const shared_ptr<Node> def_val = max_pool->get_default_value();
                const shared_ptr<op::Constant> def_const =
                    static_pointer_cast<op::Constant>(def_val);
                const vector<std::string>& values = def_const->get_value_strings();

                do_max_avg_pool_operation(topology,
                                          op->get_input_tensor_name(0),
                                          op->get_input_shape(0),
                                          op->get_output_tensor_name(0),
                                          op->get_output_shape(0),
                                          op->get_output_element_type(0),
                                          max_pool->get_window_shape(),
                                          max_pool->get_window_movement_strides(),
                                          max_pool->get_padding_below(),
                                          false,
                                          values.at(0),
                                          true);
            }
            else
            {
                do_pooling_operation(topology,
                                     op,
                                     max_pool->get_window_shape(),
                                     max_pool->get_window_movement_strides(),
                                     max_pool->get_padding_below(),
                                     cldnn::pooling_mode::max);
            }
            break;
        }
        case OP_TYPEID::MaxPoolBackprop:
        {
            if (op->get_input_size() == 3)
            {
                arguments_check(op, 3, 1);
            }
            else
            {
                arguments_check(op, 2, 1);
            }

            const shared_ptr<op::MaxPoolBackprop> max_pool_b =
                static_pointer_cast<op::MaxPoolBackprop>(op);

            do_max_pool_backprop_operation(topology,
                                           op->get_input_tensor_name(0),
                                           op->get_input_shape(0),
                                           op->get_input_tensor_name(1),
                                           op->get_input_shape(1),
                                           op->get_output_tensor_name(0),
                                           op->get_output_shape(0),
                                           op->get_output_element_type(0),
                                           max_pool_b->get_window_shape(),
                                           max_pool_b->get_window_movement_strides(),
                                           max_pool_b->get_padding_below());
            break;
        }
        case OP_TYPEID::AvgPool:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::AvgPool> avg_pool = static_pointer_cast<op::AvgPool>(op);

            if ((op->get_input_shape(0).size() > 4) ||
                (op->get_output_element_type(0) != element::f32) ||
                avg_pool->get_include_padding_in_avg_computation() ||
                has_non_zero(avg_pool->get_padding_below()) ||
                has_non_zero(avg_pool->get_padding_above()))
            {
                const shared_ptr<Node> def_val = avg_pool->get_default_value();
                const shared_ptr<op::Constant> def_const =
                    static_pointer_cast<op::Constant>(def_val);
                const vector<std::string>& values = def_const->get_value_strings();

                do_max_avg_pool_operation(topology,
                                          op->get_input_tensor_name(0),
                                          op->get_input_shape(0),
                                          op->get_output_tensor_name(0),
                                          op->get_output_shape(0),
                                          op->get_output_element_type(0),
                                          avg_pool->get_window_shape(),
                                          avg_pool->get_window_movement_strides(),
                                          avg_pool->get_padding_below(),
                                          avg_pool->get_include_padding_in_avg_computation(),
                                          values.at(0),
                                          false);
            }
            else
            {
                const cldnn::pooling_mode mode = avg_pool->get_include_padding_in_avg_computation()
                                                     ? cldnn::pooling_mode::average
                                                     : cldnn::pooling_mode::average_no_padding;

                do_pooling_operation(topology,
                                     op,
                                     avg_pool->get_window_shape(),
                                     avg_pool->get_window_movement_strides(),
                                     avg_pool->get_padding_below(),
                                     mode);
            }
            break;
        }
        case OP_TYPEID::AvgPoolBackprop:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::AvgPoolBackprop> avg_pool_b =
                static_pointer_cast<op::AvgPoolBackprop>(op);

            do_avg_pool_backprop_operation(topology,
                                           op->get_input_tensor_name(0),
                                           op->get_input_shape(0),
                                           op->get_output_tensor_name(0),
                                           op->get_output_shape(0),
                                           op->get_output_element_type(0),
                                           avg_pool_b->get_window_shape(),
                                           avg_pool_b->get_window_movement_strides(),
                                           avg_pool_b->get_padding_below(),
                                           avg_pool_b->get_include_padding_in_avg_computation());
            break;
        }
        case OP_TYPEID::Broadcast:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Broadcast> broadcast = static_pointer_cast<op::Broadcast>(op);
            const AxisSet& axis = broadcast->get_broadcast_axes();

            if (axis.empty())
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else if ((op->get_output_shape(0).size() <= 4) &&
                     (shape_size(op->get_output_shape(0)) > 0) &&
                     ((op->get_input_element_type(0) == element::f32) ||
                      (op->get_input_element_type(0) == element::i32)))
            {
                const size_t shift = 4 - op->get_output_shape(0).size();
                vector<uint16_t> fixed_b_axes;

                for (uint16_t i = 0; i < shift; ++i)
                {
                    fixed_b_axes.push_back(i);
                }

                for (auto it = axis.cbegin(); it != axis.cend(); ++it)
                {
                    fixed_b_axes.push_back(*it + shift);
                }

                const cldnn::tensor output_tensor_size =
                    intelgpu_space::create_cldnn_tensor(op->get_output_shape(0));

                const cldnn::broadcast cldnn_broadcast(op->get_output_tensor_name(0),
                                                       op->get_input_tensor_name(0),
                                                       output_tensor_size,
                                                       fixed_b_axes);
                topology.add(cldnn_broadcast);
            }
            else
            {
                kern.emit<op::Broadcast>(broadcast);
            }
            break;
        }
        case OP_TYPEID::Sum:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Sum> sum = static_pointer_cast<op::Sum>(op);
            const AxisSet& axis = sum->get_reduction_axes();

            if (axis.empty())
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else
            {
                kern.emit<op::Sum>(sum);
            }
            break;
        }
        case OP_TYPEID::Product:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Product> prod = static_pointer_cast<op::Product>(op);
            const AxisSet& axis = prod->get_reduction_axes();

            if (axis.empty())
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else
            {
                kern.emit<op::Product>(prod);
            }
            break;
        }
        case OP_TYPEID::Reshape:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Reshape> op_reshape = static_pointer_cast<op::Reshape>(op);
            const AxisVector& reshape_axes = op_reshape->get_input_order();

            if ((op->get_input_element_type(0) != element::f32) ||
                (op->get_input_shape(0).size() > 4) || (op->get_output_shape(0).size() > 4))
            {
                do_reshape_operation(topology,
                                     op->get_input_tensor_name(0),
                                     op->get_input_shape(0),
                                     op->get_input_element_type(0),
                                     op->get_output_tensor_name(0),
                                     op->get_output_shape(0),
                                     op->get_output_element_type(0),
                                     reshape_axes);
            }
            else
            {
                if (op_reshape->get_is_transpose())
                {
                    vector<uint16_t> permute_order({0, 1, 2, 3}); // No action by default
                    const size_t max_dim = 4;
                    const size_t scale =
                        reshape_axes.size() < max_dim ? max_dim - reshape_axes.size() : 0;

                    // Need to scale indexes up according on array rank.
                    // For example, in 2D array, indexes are 0,1 but in 4D array it should be 2,3
                    // because cldnn::tensor is always 4D assuming cldnn::bfyx model
                    size_t rindex = max_dim;
                    for (auto i = reshape_axes.crbegin(); i != reshape_axes.crend() && rindex > 0;
                         ++i, --rindex)
                    {
                        permute_order.at(rindex - 1) = *i + scale;
                    }

                    const cldnn::permute cldnn_permute(
                        op->get_output_tensor_name(0), op->get_input_tensor_name(0), permute_order);
                    topology.add(cldnn_permute);
                }
                else
                {
                    const cldnn::tensor new_shape =
                        intelgpu_space::create_cldnn_tensor(op->get_output_shape(0));
                    const cldnn::reshape reshape_op(
                        op->get_output_tensor_name(0), op->get_input_tensor_name(0), new_shape);
                    topology.add(reshape_op);
                }
            }
            break;
        }
        case OP_TYPEID::All:
        {
            arguments_check(op, 1, 1);

            // Empty axis is not a case for do_equal_propagation()
            kern.emit<op::All>(static_pointer_cast<op::All>(op));
            break;
        }
        case OP_TYPEID::Any:
        {
            arguments_check(op, 1, 1);

            // Empty axis is not a case for do_equal_propagation()
            kern.emit<op::Any>(static_pointer_cast<op::Any>(op));
            break;
        }
        case OP_TYPEID::ReluBackprop:
        {
            arguments_check(op, 2, 1);

            if (op->get_input_element_type(0) != element::f32 ||
                op->get_input_element_type(1) != element::f32 ||
                op->get_output_element_type(0) != element::f32 ||
                op->get_output_shape(0).size() > 4)
            {
                do_relu_backprop(topology,
                                 op->get_input_tensor_name(0),
                                 op->get_input_shape(0),
                                 op->get_input_element_type(0),
                                 op->get_input_tensor_name(1),
                                 op->get_input_shape(1),
                                 op->get_output_tensor_name(0),
                                 op->get_output_shape(0),
                                 op->get_output_element_type(0));
            }
            else
            {
                const cldnn_activation_additional_params& param = {0.f, 0.f};
                const cldnn::activation_grad cldnn_activ_grad(op->get_output_tensor_name(0),
                                                              op->get_input_tensor_name(1),
                                                              op->get_input_tensor_name(0),
                                                              activation_grad_relu,
                                                              param);
                topology.add(cldnn_activ_grad);
            }
            break;
        }
        case OP_TYPEID::Abs:
        {
            do_universal_unary(topology, op, "fabs(input_var)", activation_abs);
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            do_universal_unary(topology, op, "sqrt(input_var)", activation_sqrt);
            break;
        }
        case OP_TYPEID::Tanh:
        {
            do_universal_unary(topology, op, "tanh(input_var)", activation_hyperbolic_tan);
            break;
        }
        case OP_TYPEID::Sin:
        {
            do_universal_unary(topology, op, "sin(input_var)", activation_sin);
            break;
        }
        case OP_TYPEID::Asin:
        {
            do_universal_unary(topology, op, "asin(input_var)", activation_asin);
            break;
        }
        case OP_TYPEID::Sinh:
        {
            do_universal_unary(topology, op, "sinh(input_var)", activation_sinh);
            break;
        }
        case OP_TYPEID::Cos:
        {
            do_universal_unary(topology, op, "cos(input_var)", activation_cos);
            break;
        }
        case OP_TYPEID::Acos:
        {
            do_universal_unary(topology, op, "acos(input_var)", activation_acos);
            break;
        }
        case OP_TYPEID::Cosh:
        {
            do_universal_unary(topology, op, "cosh(input_var)", activation_cosh);
            break;
        }
        case OP_TYPEID::Log:
        {
            // clDNN doesn't provide required accuracy
            do_universal_unary(topology, op, "log(input_var)", activation_log, true);
            break;
        }
        case OP_TYPEID::Exp:
        {
            do_universal_unary(topology, op, "exp(input_var)", activation_exp);
            break;
        }
        case OP_TYPEID::Negative:
        {
            const cldnn_activation_additional_params param = {-1.f, 0.f};
            do_universal_unary(topology, op, "-(input_var)", activation_linear, false, param);
            break;
        }
        case OP_TYPEID::Relu:
        {
            const string output_type_name = get_opencl_type_name(op->get_output_element_type(0));
            const string convert_to_type = "convert_" + output_type_name;
            const string zero_const = convert_to_type + "(0)";

            do_universal_unary(topology,
                               op,
                               "max(" + zero_const + ", " + convert_to_type + "(input_var))",
                               activation_relu_negative_slope);
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            const string one_const =
                "convert_" + get_opencl_type_name(op->get_output_element_type(0)) + "(1)";
            do_universal_unary(topology,
                               op,
                               one_const + " / (" + one_const + " + exp(-input_var))",
                               activation_logistic);
            break;
        }
        case OP_TYPEID::Atan:
        {
            do_custom_unary(topology, op, "atan(input_var)");
            break;
        }
        case OP_TYPEID::Ceiling:
        {
            if (!op->get_input_element_type(0).is_real())
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else
            {
                do_custom_unary(topology, op, "ceil(input_var)");
            }
            break;
        }
        case OP_TYPEID::Floor:
        {
            if (!op->get_input_element_type(0).is_real())
            {
                do_equal_propagation(
                    topology, op->get_input_tensor_name(0), op->get_output_tensor_name(0));
            }
            else
            {
                do_custom_unary(topology, op, "floor(input_var)");
            }
            break;
        }
        case OP_TYPEID::Sign:
        {
            do_custom_unary(topology, op, "sign(input_var)");
            break;
        }
        case OP_TYPEID::Tan:
        {
            do_custom_unary(topology, op, "tan(input_var)");
            break;
        }
        case OP_TYPEID::SigmoidBackprop:
        {
            arguments_check(op, 2, 1);

            do_sigmoid_backprop_operation(topology,
                                          op->get_input_tensor_name(0),
                                          op->get_input_shape(0),
                                          op->get_input_tensor_name(1),
                                          op->get_input_shape(1),
                                          op->get_output_tensor_name(0),
                                          op->get_output_shape(0),
                                          op->get_output_element_type(0));
            break;
        }
        case OP_TYPEID::Not:
        {
            arguments_check(op, 1, 1);

            do_not_operation(topology,
                             op->get_input_tensor_name(0),
                             op->get_input_shape(0),
                             op->get_output_tensor_name(0),
                             op->get_output_shape(0),
                             op->get_output_element_type(0));
            break;
        }
        case OP_TYPEID::Greater:
        {
            do_logical_operation<op::Greater>(kern, op);
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            do_logical_operation<op::GreaterEq>(kern, op);
            break;
        }
        case OP_TYPEID::Equal:
        {
            do_logical_operation<op::Equal>(kern, op);
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            do_logical_operation<op::NotEqual>(kern, op);
            break;
        }
        case OP_TYPEID::Less:
        {
            do_logical_operation<op::Less>(kern, op);
            break;
        }
        case OP_TYPEID::LessEq:
        {
            do_logical_operation<op::LessEq>(kern, op);
            break;
        }
        case OP_TYPEID::And:
        {
            do_logical_operation<op::And>(kern, op);
            break;
        }
        case OP_TYPEID::Or:
        {
            do_logical_operation<op::Or>(kern, op);
            break;
        }
        case OP_TYPEID::Pad:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::Pad> pad = static_pointer_cast<op::Pad>(op);
            const CoordinateDiff& pad_below = pad->get_padding_below();

            do_pad_operation(topology,
                             op->get_input_tensor_name(0),
                             op->get_input_shape(0),
                             op->get_input_tensor_name(1),
                             op->get_output_tensor_name(0),
                             op->get_output_shape(0),
                             op->get_output_element_type(0),
                             pad_below);
            break;
        }
        case OP_TYPEID::BatchNormTrainingBackprop:
        {
            arguments_check(op, 6, 3);

            kern.emit<op::BatchNormTrainingBackprop>(
                static_pointer_cast<op::BatchNormTrainingBackprop>(op));
            break;
        }
        case OP_TYPEID::BatchNormInference:
        {
            const shared_ptr<op::BatchNormInference> bnorm =
                static_pointer_cast<op::BatchNormInference>(op);
            const double eps = bnorm->get_eps_value();

            arguments_check(op, 5, 1);

            // Workaround for #2729 bug.
            // Should be removed after fix in clDNN.
            // Drop 14.0 of clDNN contains this bug.
            bool proceed_with_custom_kernel = false;
            const string& gamma = op->get_input_tensor_name(0);
            const string& beta = op->get_input_tensor_name(1);
            const string& mean = op->get_input_tensor_name(3);
            const string& variance = op->get_input_tensor_name(4);

            if ((gamma == beta) || (gamma == mean) || (gamma == variance) || (beta == mean) ||
                (beta == variance) || (mean == variance))
            {
                proceed_with_custom_kernel = true;
            }

            if (proceed_with_custom_kernel || (op->get_input_shape(2).size() != 4) ||
                (op->get_input_element_type(0) != ngraph::element::f32))
            {
                kern.emit<op::BatchNormInference>(bnorm);
            }
            else
            {
                const cldnn::batch_norm batchnorm(op->get_output_tensor_name(0),
                                                  op->get_input_tensor_name(2), // input
                                                  op->get_input_tensor_name(3), // mean
                                                  op->get_input_tensor_name(4), // variance
                                                  op->get_input_tensor_name(0), // gamma
                                                  op->get_input_tensor_name(1), // beta
                                                  eps);                         // epsilon (float)
                topology.add(batchnorm);
            }
            break;
        }
        case OP_TYPEID::BatchNormTraining:
        {
            const shared_ptr<op::BatchNormTraining> bnorm =
                static_pointer_cast<op::BatchNormTraining>(op);
            const double eps = bnorm->get_eps_value();

            if ((op->get_input_shape(2).size() != 4) ||
                (op->get_input_element_type(0) != ngraph::element::f32))
            {
                kern.emit<op::BatchNormTraining>(bnorm);
            }
            else
            {
                if (op->get_inputs().size() == 5 && op->get_outputs().size() == 1)
                {
                    const cldnn::batch_norm batchnorm(op->get_output_tensor_name(0),
                                                      op->get_input_tensor_name(2), // input
                                                      op->get_input_tensor_name(3), // mean
                                                      op->get_input_tensor_name(4), // variance
                                                      op->get_input_tensor_name(0), // gamma
                                                      op->get_input_tensor_name(1), // beta
                                                      eps); // epsilon (float)
                    topology.add(batchnorm);
                }
                else if (op->get_inputs().size() == 3 && op->get_outputs().size() == 3)
                {
                    const string mean_name = op->get_output_tensor_name(1);
                    const string variance_name = op->get_output_tensor_name(2);

                    // Create a memory for mean as mutable_data to treat it as constant
                    const cldnn::layout mean_layout = IntelGPULayout::create_cldnn_layout(
                        op->get_output_element_type(1), op->get_output_shape(1));
                    const cldnn::memory mean_mem(
                        cldnn::memory::allocate(*cldnn_engine, mean_layout));

                    const cldnn::mutable_data mean_const(mean_name, mean_mem);
                    topology.add(mean_const);

                    // Create a memory for variance as mutable_data to treat it as constant
                    const cldnn::layout variance_layout = IntelGPULayout::create_cldnn_layout(
                        op->get_output_element_type(2), op->get_output_shape(2));
                    const cldnn::memory variance_mem(
                        cldnn::memory::allocate(*cldnn_engine, variance_layout));

                    const cldnn::mutable_data variance_const(variance_name, variance_mem);
                    topology.add(variance_const);

                    const cldnn::batch_norm batchnorm(op->get_output_tensor_name(0),
                                                      op->get_input_tensor_name(2), // input
                                                      eps, // epsilon (float)
                                                      mean_name,
                                                      variance_name,
                                                      op->get_input_tensor_name(0),  // gamma
                                                      op->get_input_tensor_name(1)); // beta
                    topology.add(batchnorm);

                    // Need to mark this operation as "output" to keep mean and variance
                    // in cldnn::network
                    func_output_names.insert(op->get_output_tensor_name(0));
                }
                else
                {
                    arguments_check(op, 5, 1); // throw exception in this case
                }
            }
            break;
        }
        case OP_TYPEID::Convolution:
        case OP_TYPEID::ConvolutionBias:
        case OP_TYPEID::ConvolutionBiasAdd:
        case OP_TYPEID::GroupConvolution:
        {
            // since bad inheritance design of these classes
            Strides win_stride;
            Strides win_dilation;
            Strides data_dilation;
            CoordinateDiff pad_below;
            CoordinateDiff pad_above;

            if (op_type_id == OP_TYPEID::ConvolutionBias)
            {
                arguments_check(op, 3, 1);

                const shared_ptr<op::ConvolutionBias> conv_op =
                    static_pointer_cast<op::ConvolutionBias>(op);
                win_stride = conv_op->get_window_movement_strides();
                win_dilation = conv_op->get_window_dilation_strides();
                data_dilation = conv_op->get_data_dilation_strides();
                pad_below = conv_op->get_padding_below();
                pad_above = conv_op->get_padding_above();
            }
            else if (op_type_id == OP_TYPEID::ConvolutionBiasAdd)
            {
                arguments_check(op, 4, 1);

                const shared_ptr<op::ConvolutionBiasAdd> conv_op =
                    static_pointer_cast<op::ConvolutionBiasAdd>(op);
                win_stride = conv_op->get_window_movement_strides();
                win_dilation = conv_op->get_window_dilation_strides();
                data_dilation = conv_op->get_data_dilation_strides();
                pad_below = conv_op->get_padding_below();
                pad_above = conv_op->get_padding_above();
            }
            else if (op_type_id == OP_TYPEID::GroupConvolution)
            {
                arguments_check(op, 2, 1);

                const shared_ptr<op::GroupConvolution> conv_op =
                    static_pointer_cast<op::GroupConvolution>(op);
                win_stride = conv_op->get_window_movement_strides();
                win_dilation = conv_op->get_window_dilation_strides();
                data_dilation = conv_op->get_data_dilation_strides();
                pad_below = conv_op->get_padding_below();
                pad_above = conv_op->get_padding_above();
            }
            else
            {
                arguments_check(op, 2, 1);

                const shared_ptr<op::Convolution> conv_op =
                    static_pointer_cast<op::Convolution>(op);
                win_stride = conv_op->get_window_movement_strides();
                win_dilation = conv_op->get_window_dilation_strides();
                data_dilation = conv_op->get_data_dilation_strides();
                pad_below = conv_op->get_padding_below();
                pad_above = conv_op->get_padding_above();
            }

            // clDNN has quite limited support for Convolution operation
            // following are the checks to go with workaround
            if ((win_stride.size() != 2) || (pad_below.size() != 2) || (pad_above.size() != 2) ||
                (win_dilation.size() != 2) || (data_dilation.size() != 2) ||
                (data_dilation.at(0) != 1) || (data_dilation.at(1) != 1) ||
                (op->get_output_element_type(0) != element::f32))
            {
                if (op_type_id == OP_TYPEID::ConvolutionBias)
                {
                    kern.emit<op::ConvolutionBias>(static_pointer_cast<op::ConvolutionBias>(op));
                }
                else if (op_type_id == OP_TYPEID::ConvolutionBiasAdd)
                {
                    kern.emit<op::ConvolutionBiasAdd>(
                        static_pointer_cast<op::ConvolutionBiasAdd>(op));
                }
                else if (op_type_id == OP_TYPEID::GroupConvolution)
                {
                    kern.emit<op::GroupConvolution>(static_pointer_cast<op::GroupConvolution>(op));
                }
                else
                {
                    kern.emit<op::Convolution>(static_pointer_cast<op::Convolution>(op));
                }
            }
            else
            {
                cldnn::tensor::value_type input_offset_x = -pad_below.at(1);
                cldnn::tensor::value_type input_offset_y = -pad_below.at(0);
                std::string op_input_name = op->get_input_tensor_name(0);

                if ((pad_below.at(0) != pad_above.at(0)) || (pad_below.at(1) != pad_above.at(1)))
                {
                    // Different input padding for operation workarounded by adding aux layer
                    const cldnn::tensor border_pad_above(0, 0, pad_below.at(1), pad_below.at(0), 0);
                    const cldnn::tensor border_pad_below(0, 0, pad_above.at(1), pad_above.at(0), 0);
                    input_offset_x = 0;
                    input_offset_y = 0;
                    op_input_name =
                        op_input_name + "_" + op->get_output_tensor_name(0) + "_bordered";

                    const cldnn::border cldnn_border(op_input_name,
                                                     op->get_input_tensor_name(0),
                                                     border_pad_above,
                                                     border_pad_below,
                                                     cldnn::border_type::zero);
                    topology.add(cldnn_border);
                }

                const cldnn::tensor input_offset(0, 0, input_offset_x, input_offset_y, 0);
                const cldnn::tensor strides(1, 1, win_stride.at(1), win_stride.at(0));
                const cldnn::tensor dilation(1, 1, win_dilation.at(1), win_dilation.at(0));

                if (op_type_id == OP_TYPEID::ConvolutionBias)
                {
                    const cldnn::convolution cldnn_conv_bias(op->get_output_tensor_name(0),
                                                             op_input_name,
                                                             {op->get_input_tensor_name(1)},
                                                             {op->get_input_tensor_name(2)},
                                                             strides,
                                                             input_offset,
                                                             dilation);
                    topology.add(cldnn_conv_bias);
                }
                else if (op_type_id == OP_TYPEID::ConvolutionBiasAdd)
                {
                    // Do not understand which cldnn::convolution::ctor() should be called
                    // make it clear by two operations
                    const string intermediate_name =
                        op_input_name + op->get_output_tensor_name(0) + "_intermediate";

                    const cldnn::convolution cldnn_conv_bias(intermediate_name,
                                                             op_input_name,
                                                             {op->get_input_tensor_name(1)},
                                                             {op->get_input_tensor_name(2)},
                                                             strides,
                                                             input_offset,
                                                             dilation);
                    topology.add(cldnn_conv_bias);

                    const cldnn::eltwise cldnn_conv_bias_add(
                        op->get_output_tensor_name(0),
                        {intermediate_name, op->get_input_tensor_name(3)},
                        cldnn::eltwise_mode::sum);

                    topology.add(cldnn_conv_bias_add);
                }
                else if (op_type_id == OP_TYPEID::GroupConvolution)
                {
                    const shared_ptr<op::GroupConvolution> conv_op =
                        static_pointer_cast<op::GroupConvolution>(op);

                    const cldnn::convolution cldnn_conv(op->get_output_tensor_name(0),
                                                        op_input_name,
                                                        {op->get_input_tensor_name(1)},
                                                        conv_op->get_groups(),
                                                        strides,
                                                        input_offset,
                                                        dilation);
                    topology.add(cldnn_conv);
                }
                else
                {
                    const cldnn::convolution cldnn_conv(op->get_output_tensor_name(0),
                                                        op_input_name,
                                                        {op->get_input_tensor_name(1)},
                                                        strides,
                                                        input_offset,
                                                        dilation);
                    topology.add(cldnn_conv);
                }
            }
            break;
        }
        case OP_TYPEID::ConvolutionBiasBackpropFiltersBias:
        {
            arguments_check(op, 2, 2);

            kern.emit<op::ConvolutionBiasBackpropFiltersBias>(
                static_pointer_cast<op::ConvolutionBiasBackpropFiltersBias>(op));
            break;
        }
        case OP_TYPEID::ConvolutionBackpropFilters:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::ConvolutionBackpropFilters> conv_op =
                static_pointer_cast<op::ConvolutionBackpropFilters>(op);

            const Strides& win_stride = conv_op->get_window_dilation_strides_forward();
            const CoordinateDiff& pad_below = conv_op->get_padding_below_forward();
            CoordinateDiff pad_above = conv_op->compute_backward_in_pad_above();
            const Strides& win_dilation = conv_op->get_window_movement_strides_forward();
            const Strides& data_dilation = conv_op->get_data_dilation_strides_forward();

            // workaround to use custom kernel in case of filter output NxCx1x1
            bool proceed_with_custom_kernel = false;
            const Shape& output_shape = op->get_output_shape(0);
            if ((output_shape.size() == 4) && (output_shape.at(2) == 1) &&
                (output_shape.at(3) == 1))
            {
                proceed_with_custom_kernel = true;
            }

            if ((win_stride.size() != 2) || (win_stride.at(0) != 1) || (win_stride.at(1) != 1) ||
                (pad_below.size() != 2) || (pad_above.size() != 2) || (data_dilation.size() != 2) ||
                (data_dilation.at(0) != 1) || (data_dilation.at(1) != 1) ||
                (win_dilation.size() != 2) || (op->get_output_element_type(0) != element::f32) ||
                proceed_with_custom_kernel)
            {
                kern.emit<op::ConvolutionBackpropFilters>(conv_op);
            }
            else
            {
                cldnn::tensor::value_type pad_above_x = pad_below.at(1);
                cldnn::tensor::value_type pad_above_y = pad_below.at(0);
                cldnn::tensor::value_type pad_below_x = pad_above.at(1);
                cldnn::tensor::value_type pad_below_y = pad_above.at(0);

                if ((win_dilation.at(0) != 1) && (win_dilation.at(1) != 1) &&
                    (pad_below.at(0) != pad_above.at(0)) && (pad_below.at(1) != pad_above.at(1)))
                {
                    pad_below_x += 1;
                    pad_below_y += 1;
                }

                cldnn::tensor::value_type input_offset_x = -pad_above_x;
                cldnn::tensor::value_type input_offset_y = -pad_above_y;

                std::string op_input_name = op->get_input_tensor_name(0);

                string filter_name = op->get_output_tensor_name(0) + "_filter_output";

                // Create a memory for filter as mutable_data to treat it as constant
                const cldnn::layout filter_layout = IntelGPULayout::create_cldnn_layout(
                    op->get_output_element_type(0), op->get_output_shape(0));

                const cldnn::memory filter_mem(
                    cldnn::memory::allocate(*cldnn_engine, filter_layout));

                const cldnn::mutable_data filter_const(filter_name, filter_mem);
                topology.add(filter_const);

                if ((pad_below_x != pad_above_x) && (pad_below_y != pad_above_y))
                {
                    // Different input padding for operation workarounded by adding aux layer
                    const cldnn::tensor border_pad_above(0, 0, pad_above_x, pad_above_y, 0);
                    const cldnn::tensor border_pad_below(0, 0, pad_below_x, pad_below_y, 0);
                    input_offset_x = 0;
                    input_offset_y = 0;
                    op_input_name =
                        op_input_name + "_" + op->get_output_tensor_name(0) + "_bordered";
                    const cldnn::border cldnn_border(op_input_name,
                                                     op->get_input_tensor_name(0),
                                                     border_pad_above,
                                                     border_pad_below,
                                                     cldnn::border_type::zero);
                    topology.add(cldnn_border);
                }

                const cldnn::tensor input_offset(0, 0, input_offset_x, input_offset_y, 0);
                const cldnn::tensor strides(1, 1, win_dilation.at(1), win_dilation.at(0));

                const cldnn::convolution_grad_weights conv_back_flt(op->get_output_tensor_name(0),
                                                                    op->get_input_tensor_name(1),
                                                                    op_input_name,
                                                                    {filter_name},
                                                                    strides,
                                                                    input_offset,
                                                                    cldnn::tensor(1, 1, 1, 1),
                                                                    true);
                topology.add(conv_back_flt);
            }
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::ConvolutionBackpropData> conv_op =
                static_pointer_cast<op::ConvolutionBackpropData>(op);
            const Strides& win_stride = conv_op->get_data_dilation_strides_forward();
            CoordinateDiff pad_below = conv_op->compute_backward_delta_out_pad_below();
            CoordinateDiff pad_above = conv_op->compute_backward_delta_out_pad_above();
            const Strides& win_dilation = conv_op->get_window_dilation_strides_forward();
            const Strides& data_dilation = conv_op->get_window_movement_strides_forward();

            if ((win_stride.size() != 2) || (win_stride.at(0) != 1) || (win_stride.at(1) != 1) ||
                (pad_below.size() != 2) || (pad_above.size() != 2) || (data_dilation.size() != 2) ||
                (data_dilation.at(0) != 1) || (data_dilation.at(1) != 1) ||
                (win_dilation.size() != 2) || (win_dilation.at(0) != 1) ||
                (win_dilation.at(1) != 1) || (op->get_output_element_type(0) != element::f32) ||
                ((pad_below.at(0) == pad_above.at(0)) && (pad_below.at(1) == pad_above.at(1))))
            {
                kern.emit<op::ConvolutionBackpropData>(conv_op);
            }
            else
            {
                cldnn::tensor::value_type input_offset_xy = -1;
                std::string op_input_name = op->get_input_tensor_name(1);

                if ((pad_below.at(0) == pad_above.at(0)) && (pad_below.at(1) == pad_above.at(1)))
                {
                    // symmetric padding case temporally excluded (custom kernel executed) due to stability issues
                    const CoordinateDiff& pad_below_for = conv_op->get_padding_below_forward();
                    input_offset_xy = -pad_below_for.at(0);
                }
                else
                {
                    // Different input padding for operation workarounded by adding aux layer
                    const cldnn::tensor crop_pad_below(0, 0, -pad_below.at(1), -pad_below.at(0), 0);
                    const cldnn::tensor crop_pad_above(0, 0, -pad_above.at(1), -pad_above.at(0), 0);
                    op_input_name =
                        op_input_name + "_" + op->get_output_tensor_name(0) + "_cropped";

                    const cldnn::crop cldnn_crop(op_input_name,
                                                 op->get_input_tensor_name(1),
                                                 crop_pad_below,
                                                 crop_pad_above,
                                                 cldnn::crop_borders_t());
                    topology.add(cldnn_crop);
                }

                const cldnn::tensor input_offset(0, 0, input_offset_xy, input_offset_xy, 0);
                const cldnn::tensor strides(1, 1, win_stride.at(1), win_stride.at(0));

                const cldnn::convolution_grad_input cldnn_conv_back_data(
                    op->get_output_tensor_name(0),
                    op_input_name,
                    {op->get_input_tensor_name(0)},
                    strides,
                    input_offset);
                topology.add(cldnn_conv_back_data);
            }
            break;
        }
        case OP_TYPEID::Min:
        {
            arguments_check(op, 1, 1);

            kern.emit<op::Min>(static_pointer_cast<op::Min>(op));
            break;
        }
        case OP_TYPEID::Max:
        {
            arguments_check(op, 1, 1);

            kern.emit<op::Max>(static_pointer_cast<op::Max>(op));
            break;
        }
        case OP_TYPEID::OneHot:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::OneHot> one_hot_op = static_pointer_cast<op::OneHot>(op);
            const size_t one_hot_axis = one_hot_op->get_one_hot_axis();

            do_one_hot_operation(topology,
                                 op->get_input_tensor_name(0),
                                 op->get_input_shape(0),
                                 op->get_input_element_type(0),
                                 op->get_output_tensor_name(0),
                                 op->get_output_shape(0),
                                 op->get_output_element_type(0),
                                 one_hot_axis);
            break;
        }
        case OP_TYPEID::ArgMax:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::ArgMax> arg_max_op = static_pointer_cast<op::ArgMax>(op);
            const size_t reduction_axis = arg_max_op->get_reduction_axis();
            const element::Type& index_elem_type = arg_max_op->get_element_type();

            if (index_elem_type == element::i64 || index_elem_type == element::i32)
            {
                do_arg_max_min_operation(topology,
                                         op->get_input_tensor_name(0),
                                         op->get_input_shape(0),
                                         op->get_input_element_type(0),
                                         op->get_output_tensor_name(0),
                                         op->get_output_shape(0),
                                         op->get_output_element_type(0),
                                         reduction_axis,
                                         true);
            }
            else
            {
                cldnn::arg_max_min::axis_name axis =
                    reduction_axis == 0 ? cldnn::arg_max_min::y : cldnn::arg_max_min::x;
                const cldnn::arg_max_min arg_max_min(op->get_output_tensor_name(0),
                                                     op->get_input_tensor_name(0),
                                                     cldnn::arg_max_min::max,
                                                     1,
                                                     axis);
                topology.add(arg_max_min);
            }
            break;
        }
        case OP_TYPEID::ArgMin:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::ArgMin> arg_min_op = static_pointer_cast<op::ArgMin>(op);
            const size_t reduction_axis = arg_min_op->get_reduction_axis();
            const element::Type& index_elem_type = arg_min_op->get_element_type();

            if (index_elem_type == element::i64 || index_elem_type == element::i32)
            {
                do_arg_max_min_operation(topology,
                                         op->get_input_tensor_name(0),
                                         op->get_input_shape(0),
                                         op->get_input_element_type(0),
                                         op->get_output_tensor_name(0),
                                         op->get_output_shape(0),
                                         op->get_output_element_type(0),
                                         reduction_axis,
                                         false);
            }
            else
            {
                cldnn::arg_max_min::axis_name axis =
                    reduction_axis == 0 ? cldnn::arg_max_min::y : cldnn::arg_max_min::x;
                const cldnn::arg_max_min arg_max_min(op->get_output_tensor_name(0),
                                                     op->get_input_tensor_name(0),
                                                     cldnn::arg_max_min::min,
                                                     1,
                                                     axis);
                topology.add(arg_max_min);
            }
            break;
        }
        case OP_TYPEID::Quantize:
        {
            arguments_check(op, 3, 1);

            const shared_ptr<op::Quantize> quant_op = static_pointer_cast<op::Quantize>(op);
            const AxisSet& axes = quant_op->get_axes();
            const op::Quantize::RoundMode mode = quant_op->get_round_mode();

            do_quantize_operation(topology,
                                  op->get_input_tensor_name(0),
                                  op->get_input_shape(0),
                                  op->get_input_element_type(0),
                                  op->get_input_tensor_name(1),
                                  op->get_input_shape(1),
                                  op->get_input_tensor_name(2),
                                  op->get_input_shape(2),
                                  op->get_output_tensor_name(0),
                                  op->get_output_shape(0),
                                  op->get_output_element_type(0),
                                  axes,
                                  mode);
            break;
        }
        case OP_TYPEID::Dequantize:
        {
            arguments_check(op, 3, 1);

            const shared_ptr<op::Dequantize> dequ_op = static_pointer_cast<op::Dequantize>(op);
            const AxisSet& axes = dequ_op->get_axes();

            do_dequantize_operation(topology,
                                    op->get_input_tensor_name(0),
                                    op->get_input_shape(0),
                                    op->get_input_element_type(0),
                                    op->get_input_tensor_name(1),
                                    op->get_input_shape(1),
                                    op->get_input_element_type(1),
                                    op->get_input_tensor_name(2),
                                    op->get_input_shape(2),
                                    op->get_input_element_type(2),
                                    op->get_output_tensor_name(0),
                                    op->get_output_shape(0),
                                    op->get_output_element_type(0),
                                    axes);
            break;
        }
        case OP_TYPEID::LRN:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::LRN> lrn_op = static_pointer_cast<op::LRN>(op);

            const cldnn::lrn lrn(op->get_output_tensor_name(0),
                                 op->get_input_tensor_name(0),
                                 lrn_op->get_nsize(),
                                 lrn_op->get_bias(),
                                 lrn_op->get_alpha(),
                                 lrn_op->get_beta(),
                                 cldnn_lrn_norm_region_across_channel);
            topology.add(lrn);
            break;
        }
        case OP_TYPEID::TopK:
        {
            arguments_check(op, 1, 2);

            const shared_ptr<op::TopK> topk_op = static_pointer_cast<op::TopK>(op);

            const size_t top_k_axis = topk_op->get_top_k_axis();
            const element::Type& index_elem_type = topk_op->get_index_element_type();
            const size_t k = topk_op->get_k();
            const bool compute_max = topk_op->get_compute_max();

            do_topk_operation(topology,
                              op->get_input_tensor_name(0),
                              op->get_input_shape(0),
                              op->get_input_element_type(0),
                              op->get_output_tensor_name(0),
                              op->get_output_shape(0),
                              op->get_output_element_type(0),
                              index_elem_type,
                              top_k_axis,
                              k,
                              compute_max,
                              true);

            do_topk_operation(topology,
                              op->get_input_tensor_name(0),
                              op->get_input_shape(0),
                              op->get_input_element_type(0),
                              op->get_output_tensor_name(1),
                              op->get_output_shape(1),
                              op->get_output_element_type(1),
                              index_elem_type,
                              top_k_axis,
                              k,
                              compute_max,
                              false);
            break;
        }
        case OP_TYPEID::AllReduce:
        case OP_TYPEID::BatchMatMul:
        case OP_TYPEID::BroadcastDistributed:
        case OP_TYPEID::BroadcastLike:
        case OP_TYPEID::Clamp:
        case OP_TYPEID::DepthToSpace:
        case OP_TYPEID::DynBroadcast:
        case OP_TYPEID::DynPad:
        case OP_TYPEID::DynReshape:
        case OP_TYPEID::DynSlice:
        case OP_TYPEID::Elu:
        case OP_TYPEID::EmbeddingLookup:
        case OP_TYPEID::Erf:
        case OP_TYPEID::Gather:
        case OP_TYPEID::GatherND:
        case OP_TYPEID::GenerateMask:
        case OP_TYPEID::GRN:
        case OP_TYPEID::HardSigmoid:
        case OP_TYPEID::MVN:
        case OP_TYPEID::Normalize:
        case OP_TYPEID::PRelu:
        case OP_TYPEID::Passthrough:
        case OP_TYPEID::QuantizedAvgPool:
        case OP_TYPEID::QuantizedConvolution:
        case OP_TYPEID::QuantizedConvolutionBias:
        case OP_TYPEID::QuantizedConvolutionBiasAdd:
        case OP_TYPEID::QuantizedConvolutionBiasSignedAdd:
        case OP_TYPEID::QuantizedConvolutionRelu:
        case OP_TYPEID::QuantizedDot:
        case OP_TYPEID::QuantizedDotBias:
        case OP_TYPEID::QuantizedMaxPool:
        case OP_TYPEID::ReplaceSlice:
        case OP_TYPEID::ScalarConstantLike:
        case OP_TYPEID::ScaleShift:
        case OP_TYPEID::ScatterAdd:
        case OP_TYPEID::ScatterNDAdd:
        case OP_TYPEID::ShapeOf:
        case OP_TYPEID::SpaceToDepth:
        case OP_TYPEID::Squeeze:
        case OP_TYPEID::StopGradient:
        case OP_TYPEID::Tile:
        case OP_TYPEID::Transpose:
        case OP_TYPEID::Unsqueeze:
        default:
        {
            throw unsupported_op("Unsupported op '" + op->description() +
                                 "' in IntelGPU back end.");
        }
#pragma GCC diagnostic pop
        }
    }

    cldnn::build_options network_build_options;

    network_build_options.set_option(cldnn::build_option::optimize_data(m_cldnn_graph_optimize));

    if (!func_output_names.empty())
    {
        vector<cldnn::primitive_id> names_vec(func_output_names.begin(), func_output_names.end());
        network_build_options.set_option(cldnn::build_option::outputs(names_vec));
    }

    if (m_cldnn_dump_enable)
    {
        network_build_options.set_option(cldnn::build_option::graph_dumps_dir(m_cldnn_dump_dir));
    }

    shared_ptr<cldnn::network> cldnn_network =
        make_shared<cldnn::network>(*cldnn_engine, topology, network_build_options);

    if (m_profile_enable)
    {
        timer_compile.stop();
        compilation_time = timer_compile.get_milliseconds();
        consumed_memory = runtime::intelgpu::get_max_memory_rss() - consumed_memory;
    }

    rc = make_shared<IntelGPUExecutable>(func,
                                         cldnn_network,
                                         enable_timing,
                                         m_profile_enable,
                                         compilation_time,
                                         consumed_memory,
                                         m_profile_lines_limit_count);
    if (!m_function_cache_disabled)
    {
        cldnn_networks.insert({func, rc});
    }

    return rc;
}

void runtime::intelgpu::IntelGPUBackend::remove_compiled_function(shared_ptr<Executable> exec)
{
    for (auto it = cldnn_networks.begin(); it != cldnn_networks.end(); ++it)
    {
        if (it->second == exec)
        {
            cldnn_networks.erase(it);
            break;
        }
    }
}

bool runtime::intelgpu::IntelGPUBackend::is_supported_property(const Property prop) const
{
    if (prop == Property::memory_attach)
    {
        return true;
    }

    return false;
}

bool runtime::intelgpu::IntelGPUBackend::is_supported(const Node& node) const
{
    return is_supported_impl(node);
}

bool runtime::intelgpu::IntelGPUBackend::is_supported_impl(const Node& node)
{
    const OP_TYPEID op_type_id = get_typeid(node.description());
    switch (op_type_id)
    {
    case OP_TYPEID::Clamp:
    case OP_TYPEID::HardSigmoid:
    case OP_TYPEID::DepthToSpace:
    case OP_TYPEID::Elu:
    case OP_TYPEID::Gemm:
    case OP_TYPEID::GRN:
    case OP_TYPEID::MVN:
    case OP_TYPEID::Normalize:
    case OP_TYPEID::PRelu:
    case OP_TYPEID::ScaleShift:
    case OP_TYPEID::SpaceToDepth:
    case OP_TYPEID::Squeeze:
    case OP_TYPEID::Unsqueeze: { return false;
    }
    default: { return true;
    }
    }
}
