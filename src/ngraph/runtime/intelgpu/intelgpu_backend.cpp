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

#include <iomanip>
#include <sys/resource.h>
#include <sys/time.h>

#include <CPP/activation.hpp>
#include <CPP/activation_grad.hpp>
#include <CPP/arg_max_min.hpp>
#include <CPP/batch_norm.hpp>
#include <CPP/border.hpp>
#include <CPP/broadcast.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/convolution.hpp>
#include <CPP/convolution_grad_input.hpp>
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
#include <CPP/reorder.hpp>
#include <CPP/reshape.hpp>
#include <CPP/scale.hpp>
#include <CPP/select.hpp>
#include <CPP/softmax.hpp>
#include <CPP/topology.hpp>

#include "ngraph/pass/algebraic_simplification.hpp"
#include "ngraph/pass/any_all_replacement.hpp"
#include "ngraph/pass/cse.hpp"
#include "ngraph/pass/get_output_element_elimination.hpp"
#include "ngraph/pass/manager.hpp"
#include "ngraph/pass/nop_elimination.hpp"
#include "ngraph/pass/reshape_elimination.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_batchnorm.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_broadcast.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_convolution.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_func_call.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_softmax.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"
#include "ngraph/runtime/intelgpu/visualize_tree.hpp"

#include "ngraph/file_util.hpp"
#include "ngraph/function.hpp"
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
#include "ngraph/op/embedding_lookup.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/lrn.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reduce.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/parameter_vector.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

using intelgpu_space = runtime::intelgpu::IntelGPULayout;

#define USE_INTELGPU_CUSTOM_KERNELS 0

// This expands the op list in op_tbl.hpp into a list of enumerations that look like this:
// Abs,
// Acos,
// ...
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

static void arguments_check(const shared_ptr<Node>& op, size_t input, size_t output)
{
    if (op->get_input_size() != input || op->get_output_size() != output)
    {
        ostringstream os;
        os << "Operation \"" << op->description() << "\" input and output sizes mismatch."
           << " Expected input size=" << input << ", provided=" << op->get_input_size()
           << ". Expected output size=" << output << ", provided=" << op->get_output_size();
        throw invalid_argument(os.str());
    }
}

static void
    memory_size_check(size_t memory_size, const shared_ptr<Node>& node, const string& function_name)
{
    const size_t tensor_size = shape_size(node->get_shape()) * node->get_element_type().size();

    if (memory_size != tensor_size)
    {
        ostringstream os;
        os << "IntelGPU backend failed memory check. In \"" << function_name << "\" with Node \""
           << node->get_name() << "\" and " << node->get_shape() << " mismatched memory sizes "
           << tensor_size << " and " << memory_size;
        throw invalid_argument(os.str());
    }
}

static const string& get_input_name(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_tensor().get_name();
}

static const string& get_output_name(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_tensor().get_name();
}

static const Shape& get_input_shape(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_shape();
}

static const Shape& get_output_shape(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_shape();
}

static const element::Type& get_input_type(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_inputs().at(num).get_tensor().get_element_type();
}

static const element::Type& get_output_type(const shared_ptr<Node>& op, size_t num = 0)
{
    return op->get_outputs().at(num).get_tensor().get_element_type();
}

static void do_eltwise_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 cldnn::eltwise_mode mode)
{
    arguments_check(op, 2, 1);

    if ((get_input_type(op) == element::i32 || get_input_type(op) == element::i64) &&
        (mode == cldnn::eltwise_mode::min || mode == cldnn::eltwise_mode::max))
    {
        string custom_op;

        if (mode == cldnn::eltwise_mode::min)
        {
            custom_op = "min";
        }
        else if (mode == cldnn::eltwise_mode::max)
        {
            custom_op = "max";
        }
        else
        {
            custom_op = "not_implemented_operation";
        }

        runtime::intelgpu::do_eltwise_kernel(topology,
                                             get_input_name(op, 0),
                                             get_input_shape(op, 0),
                                             get_input_type(op, 0),
                                             get_input_name(op, 1),
                                             get_input_shape(op, 1),
                                             get_output_name(op),
                                             get_output_shape(op),
                                             get_output_type(op),
                                             custom_op);
    }
    else
    {
        const cldnn::eltwise op_add(
            get_output_name(op), {get_input_name(op, 0), get_input_name(op, 1)}, mode);
        topology.add(op_add);
    }
}

static void do_unary_operation(cldnn::topology& topology,
                               const shared_ptr<Node>& op,
                               cldnn_activation_func mode,
                               const cldnn_activation_additional_params& param = {0.f, 0.f})
{
    arguments_check(op, 1, 1);

    const cldnn::activation cldnn_unary(get_output_name(op), get_input_name(op), mode, param);
    topology.add(cldnn_unary);
}

static void do_pooling_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 const Shape& pool_shape,
                                 const Strides& pool_strides,
                                 const Shape& pad_below,
                                 const cldnn::pooling_mode mode)
{
    arguments_check(op, 1, 1);

    const cldnn::tensor output_size = intelgpu_space::create_cldnn_tensor(get_output_shape(op));
    const cldnn::tensor input_offset = intelgpu_space::create_cldnn_offset(pad_below);
    const cldnn::tensor size = intelgpu_space::create_cldnn_tensor(pool_shape);
    const cldnn::tensor stride = intelgpu_space::create_cldnn_tensor(pool_strides);

    const cldnn::pooling cldnn_pooling(
        get_output_name(op), get_input_name(op), mode, size, stride, input_offset, output_size);
    topology.add(cldnn_pooling);
}

static void do_logical_operation(cldnn::topology& topology,
                                 const shared_ptr<Node>& op,
                                 const string& operation)
{
    arguments_check(op, 2, 1);

    runtime::intelgpu::do_logic_kernel(topology,
                                       get_input_name(op, 0),
                                       get_input_shape(op, 0),
                                       get_input_type(op, 0),
                                       get_input_name(op, 1),
                                       get_input_shape(op, 1),
                                       get_output_name(op),
                                       get_output_shape(op),
                                       get_output_type(op),
                                       operation);
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

extern "C" const char* get_ngraph_version_string()
{
    return NGRAPH_VERSION;
}

extern "C" runtime::Backend* new_backend(const char* configuration_string)
{
    return new runtime::intelgpu::IntelGPUBackend();
}

extern "C" void delete_backend(runtime::Backend* backend)
{
    delete backend;
}

static size_t get_max_memory_rss()
{
    size_t result = 0;
    struct rusage usage;

    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        result = usage.ru_maxrss; // the value is in kilobytes

        // aligne result to return bytes
        result *= 1000;
    }

    return result;
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
    if (getenv("NGRAPH_INTELGPU_DISABLE_OPTIMIZATIONS") != nullptr)
    {
        m_disable_backend_optimizations = true;
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

    cldnn::engine_configuration cldnn_configuration(profiling);
    ocl_engine = make_shared<cldnn::engine>(cldnn_configuration);
}

shared_ptr<runtime::Tensor>
    runtime::intelgpu::IntelGPUBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(element_type, shape, *ocl_engine);
}

shared_ptr<runtime::Tensor> runtime::intelgpu::IntelGPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *ocl_engine, memory_pointer);
}

runtime::Handle runtime::intelgpu::IntelGPUBackend::compile(shared_ptr<Function> func)
{
    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network != nullptr)
    {
        return func;
    }

    set<cldnn::primitive_id> func_output_names;
    cldnn::topology topology;

    if (m_dump_graph_enable)
    {
        visualize_tree(func, "intelgpu_", "_orig");
    }

    if (!m_disable_backend_optimizations)
    {
        ngraph::pass::Manager pass_manager;

        pass_manager.register_pass<ngraph::pass::NopElimination>();
        pass_manager.register_pass<ngraph::pass::AlgebraicSimplification>();
        pass_manager.register_pass<ngraph::pass::CommonSubexpressionElimination>();
        pass_manager.register_pass<ngraph::pass::ReshapeElimination>();

        // GetOutputElementElimination must be after CommonSubexpressionElimination
        pass_manager.register_pass<ngraph::pass::GetOutputElementElimination>();

        pass_manager.run_passes(func);

        if (m_dump_graph_enable)
        {
            visualize_tree(func, "intelgpu_", "_opt");
        }
    }

    for (shared_ptr<Node> op : func->get_ops())
    {
// We want to check that every OP_TYPEID enumeration is included in the list.
// These GCC flags enable compile-time checking so that if an enumeration
// is not in the list an error is generated.
#pragma GCC diagnostic push
#pragma GCC diagnostic error "-Wswitch"
#pragma GCC diagnostic error "-Wswitch-enum"
        switch (get_typeid(op->description()))
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

            func_output_names.insert(get_input_name(op));
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

            do_equal_propagation(topology, get_input_name(op, elem->get_n()), get_output_name(op));
            break;
        }
        case OP_TYPEID::Slice:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Slice> elem = static_pointer_cast<op::Slice>(op);
            const Coordinate& lower_bounds = elem->get_lower_bounds();
            const Coordinate& upper_bounds = elem->get_upper_bounds();
            const Strides& strides = elem->get_strides();

            if (get_input_shape(op).empty() || get_output_shape(op).empty() ||
                lower_bounds.empty() || upper_bounds.empty() || strides.empty())
            {
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else
            {
                do_slice_operation(topology,
                                   get_input_name(op),
                                   get_input_shape(op),
                                   get_output_name(op),
                                   get_output_shape(op),
                                   get_output_type(op),
                                   lower_bounds,
                                   upper_bounds,
                                   strides);
            }
            break;
        }
        case OP_TYPEID::Select:
        {
            arguments_check(op, 3, 1);

// Leave it here for some time
#if USE_INTELGPU_CUSTOM_KERNELS
            do_select_operation(topology,
                                get_input_name(op, 0),
                                get_input_shape(op, 0),
                                get_input_name(op, 1),
                                get_input_shape(op, 1),
                                get_input_name(op, 2),
                                get_input_shape(op, 2),
                                get_output_name(op),
                                get_output_shape(op),
                                get_output_type(op));
#else
            const cldnn::select cldnn_select(get_output_name(op),
                                             get_input_name(op, 1),
                                             get_input_name(op, 2),
                                             get_input_name(op));
            topology.add(cldnn_select);
#endif
            break;
        }
        case OP_TYPEID::Reverse:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Reverse> reverse_op = static_pointer_cast<op::Reverse>(op);
            const AxisSet& reversed_axes = reverse_op->get_reversed_axes();

            if (reversed_axes.empty())
            {
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else
            {
                do_reverse_operation(topology,
                                     get_input_name(op),
                                     get_input_shape(op),
                                     get_output_name(op),
                                     get_output_shape(op),
                                     get_output_type(op),
                                     reversed_axes);
            }
            break;
        }
        case OP_TYPEID::Convert:
        {
            arguments_check(op, 1, 1);

            if (get_input_type(op) == get_output_type(op))
            {
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else
            {
                do_convert_operation(topology,
                                     get_input_name(op),
                                     get_input_shape(op),
                                     get_input_type(op),
                                     get_output_name(op),
                                     get_output_shape(op),
                                     get_output_type(op));
            }
            break;
        }
        case OP_TYPEID::Concat:
        {
            if (op->get_inputs().empty() || op->get_outputs().size() != 1)
            {
                arguments_check(op, 1, 1);
            }

            // All input shapes must be the same
            // if shape is empty (means Shape{}) in this case treat its size as 1
            const size_t ngraph_tensor_dims =
                get_input_shape(op).empty() ? 1 : get_input_shape(op).size();
            const shared_ptr<op::Concat> concat_op = static_pointer_cast<op::Concat>(op);
            const size_t ngraph_concat_axis = concat_op->get_concatenation_axis();
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

            const cldnn::concatenation cldnn_concat(get_output_name(op), inputs, cldnn_axis);
            topology.add(cldnn_concat);
            break;
        }
        case OP_TYPEID::Softmax:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Softmax> softmax_op = static_pointer_cast<op::Softmax>(op);
            const AxisSet& axes = softmax_op->get_axes();
            const size_t axes_size = axes.size();
            const size_t shape_dim_count = get_input_shape(op, 0).size();

            // clDNN has limited support for Softmax operation
            // following are the checks to go with custom kernel
            if ((shape_dim_count > 3) || ((shape_dim_count == 3) && (axes_size == 2)))
            {
                do_softmax_operation(topology,
                                     get_input_name(op),
                                     get_input_shape(op),
                                     get_input_type(op),
                                     get_output_name(op),
                                     get_output_shape(op),
                                     get_output_type(op),
                                     axes);
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
                    get_output_name(op), get_input_name(op), dimension);
                topology.add(cldnn_softmax);
            }
            break;
        }
        case OP_TYPEID::Add:
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::sum);
            break;
        }
        case OP_TYPEID::Multiply:
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::prod);
            break;
        }
        case OP_TYPEID::Divide:
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::div);
            break;
        }
        case OP_TYPEID::Maximum:
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::max);
            break;
        }
        case OP_TYPEID::Minimum:
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::min);
            break;
        }
        case OP_TYPEID::Constant:
        {
            arguments_check(op, 0, 1);

            const shared_ptr<op::Constant> constant_inst = static_pointer_cast<op::Constant>(op);
            void* memory_pointer = const_cast<void*>(constant_inst->get_data_ptr());

            const cldnn::layout layout =
                IntelGPULayout::create_cldnn_layout(get_output_type(op), get_output_shape(op));
            const cldnn::memory mem(
                cldnn::memory::attach<void>(layout, memory_pointer, layout.bytes_count()));

            const cldnn::data op_const(get_output_name(op), mem);
            topology.add(op_const);
            break;
        }
        case OP_TYPEID::Dot:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::Dot> dot_inst = static_pointer_cast<op::Dot>(op);
            const size_t axes_count = dot_inst->get_reduction_axes_count();
            const Shape& input0_shape = get_input_shape(op, 0);
            const Shape& input1_shape = get_input_shape(op, 1);
            const size_t input0_elem_count = shape_size(input0_shape);
            const size_t input1_elem_count = shape_size(input1_shape);

            if (get_input_type(op) == element::f32 && get_input_type(op, 1) == element::f32 &&
                get_output_type(op) == element::f32 && input0_elem_count && input1_elem_count &&
                (axes_count < 2) && (input0_shape.size() < 3) && (input1_shape.size() < 3))
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

                const cldnn::gemm dot_op(get_output_name(op),
                                         get_input_name(op, 0),
                                         get_input_name(op, 1),
                                         transpose0,
                                         transpose1);
                topology.add(dot_op);
            }
            else
            {
                do_dot_operation(topology,
                                 get_input_name(op, 0),
                                 get_input_shape(op, 0),
                                 get_input_name(op, 1),
                                 get_input_shape(op, 1),
                                 get_output_name(op),
                                 get_output_shape(op),
                                 get_output_type(op),
                                 axes_count);
            }
            break;
        }
        case OP_TYPEID::MaxPool:
        {
            const shared_ptr<op::MaxPool> max_pool = static_pointer_cast<op::MaxPool>(op);

            do_pooling_operation(topology,
                                 op,
                                 max_pool->get_window_shape(),
                                 max_pool->get_window_movement_strides(),
                                 max_pool->get_padding_below(),
                                 cldnn::pooling_mode::max);
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
                                           get_input_name(op, 0),
                                           get_input_shape(op, 0),
                                           get_input_name(op, 1),
                                           get_input_shape(op, 1),
                                           get_output_name(op),
                                           get_output_shape(op),
                                           get_output_type(op),
                                           max_pool_b->get_window_shape(),
                                           max_pool_b->get_window_movement_strides(),
                                           max_pool_b->get_padding_below());
            break;
        }
        case OP_TYPEID::AvgPool:
        {
            const shared_ptr<op::AvgPool> avg_pool = static_pointer_cast<op::AvgPool>(op);
            const cldnn::pooling_mode mode = avg_pool->get_include_padding_in_avg_computation()
                                                 ? cldnn::pooling_mode::average
                                                 : cldnn::pooling_mode::average_no_padding;

            do_pooling_operation(topology,
                                 op,
                                 avg_pool->get_window_shape(),
                                 avg_pool->get_window_movement_strides(),
                                 avg_pool->get_padding_below(),
                                 mode);
            break;
        }
        case OP_TYPEID::AvgPoolBackprop:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::AvgPoolBackprop> avg_pool_b =
                static_pointer_cast<op::AvgPoolBackprop>(op);

            do_avg_pool_backprop_operation(topology,
                                           get_input_name(op, 0),
                                           get_input_shape(op, 0),
                                           get_output_name(op),
                                           get_output_shape(op),
                                           get_output_type(op),
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
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else if (get_input_type(op) != element::i32 && get_input_type(op) != element::i64 &&
                     ((get_input_shape(op).size() == 1 && get_input_shape(op).at(0) == 1) ||
                      get_input_shape(op).empty()))
            {
                const cldnn::tensor output_tensor_size =
                    intelgpu_space::create_cldnn_tensor(get_output_shape(op));
                const cldnn::broadcast cldnn_broadcast(
                    get_output_name(op), get_input_name(op), output_tensor_size);
                topology.add(cldnn_broadcast);
            }
            else
            {
                do_bcast_sum_operation(topology,
                                       get_input_name(op),
                                       get_input_shape(op),
                                       get_input_type(op),
                                       get_output_name(op),
                                       get_output_shape(op),
                                       get_output_type(op),
                                       axis,
                                       true);
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
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else
            {
                do_bcast_sum_operation(topology,
                                       get_input_name(op),
                                       get_input_shape(op),
                                       get_input_type(op),
                                       get_output_name(op),
                                       get_output_shape(op),
                                       get_output_type(op),
                                       axis,
                                       false);
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
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else
            {
                do_product_operation(topology,
                                     get_input_name(op),
                                     get_input_shape(op),
                                     get_output_name(op),
                                     get_output_shape(op),
                                     get_output_type(op),
                                     axis);
            }
            break;
        }
        case OP_TYPEID::Reshape:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Reshape> op_reshape = static_pointer_cast<op::Reshape>(op);

            if (op_reshape->get_is_transpose())
            {
                vector<uint16_t> permute_order({0, 1, 2, 3}); // No action by default
                const AxisVector& reshape_axes = op_reshape->get_input_order();
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
                    get_output_name(op), get_input_name(op), permute_order);
                topology.add(cldnn_permute);
            }
            else
            {
                const cldnn::tensor new_shape =
                    intelgpu_space::create_cldnn_tensor(get_output_shape(op));
                const cldnn::reshape reshape_op(get_output_name(op), get_input_name(op), new_shape);
                topology.add(reshape_op);
            }
            break;
        }
        case OP_TYPEID::Negative:
        {
            if (get_input_type(op) == ngraph::element::i32)
            {
                // This is workaround to enable GNMT in training mode.
                // clDNN doesn't support i32 data type for activation primitive.
                // Exception from clDNN:  implementation_map for N5cldnn10activationE
                // could not find any implementation to match key
                do_negative_operation(topology,
                                      get_input_name(op),
                                      get_input_shape(op),
                                      get_input_type(op),
                                      get_output_name(op),
                                      get_output_shape(op),
                                      get_output_type(op));
            }
            else
            {
                const cldnn_activation_additional_params param = {-1.f, 0.f};
                do_unary_operation(topology, op, activation_linear, param);
            }
            break;
        }
        case OP_TYPEID::All:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::All> all_op = static_pointer_cast<op::All>(op);
            const AxisSet& axis = all_op->get_reduction_axes();
            const shared_ptr<Node> def_val = all_op->get_default_value();
            const shared_ptr<op::Constant> def_const = static_pointer_cast<op::Constant>(def_val);
            const vector<std::string>& values = def_const->get_value_strings();

            // Empty axis is not a case for do_equal_propagation()
            do_all_any_op(topology,
                          get_input_name(op, 0),
                          get_input_shape(op, 0),
                          get_output_name(op),
                          get_output_shape(op),
                          get_output_type(op),
                          axis,
                          "lhs && rhs",
                          values.at(0));
            break;
        }
        case OP_TYPEID::Any:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Any> any_op = static_pointer_cast<op::Any>(op);
            const AxisSet& axis = any_op->get_reduction_axes();
            const shared_ptr<Node> def_val = any_op->get_default_value();
            const shared_ptr<op::Constant> def_const = static_pointer_cast<op::Constant>(def_val);
            const vector<std::string>& values = def_const->get_value_strings();

            // Empty axis is not a case for do_equal_propagation()
            do_all_any_op(topology,
                          get_input_name(op, 0),
                          get_input_shape(op, 0),
                          get_output_name(op),
                          get_output_shape(op),
                          get_output_type(op),
                          axis,
                          "lhs || rhs",
                          values.at(0));
            break;
        }
        case OP_TYPEID::Relu:
        {
            do_unary_operation(topology, op, activation_relu);
            break;
        }
        case OP_TYPEID::ReluBackprop:
        {
            arguments_check(op, 2, 1);

            const cldnn_activation_additional_params& param = {0.f, 0.f};
            const cldnn::activation_grad cldnn_activ_grad(get_output_name(op),
                                                          get_input_name(op, 1),
                                                          get_input_name(op, 0),
                                                          activation_grad_relu,
                                                          param);
            topology.add(cldnn_activ_grad);
            break;
        }
        case OP_TYPEID::Reduce:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::Reduce> red_op = static_pointer_cast<op::Reduce>(op);
            const AxisSet& axis = red_op->get_reduction_axes();
            vector<shared_ptr<Function>> f = red_op->get_functions();

            // Empty axis is not a case for do_equal_propagation()
            do_reduce_func_call(topology,
                                get_input_name(op, 0),
                                get_input_shape(op, 0),
                                get_input_name(op, 1),
                                get_input_shape(op, 1),
                                get_output_name(op),
                                get_output_shape(op),
                                get_output_type(op),
                                axis,
                                f);
            break;
        }
        case OP_TYPEID::Abs:
        {
            do_unary_operation(topology, op, activation_abs);
            break;
        }
        case OP_TYPEID::Sqrt:
        {
            do_unary_operation(topology, op, activation_sqrt);
            break;
        }
        case OP_TYPEID::Tanh:
        {
            do_unary_operation(topology, op, activation_hyperbolic_tan);
            break;
        }
        case OP_TYPEID::Sin:
        {
            do_unary_operation(topology, op, activation_sin);
            break;
        }
        case OP_TYPEID::Asin:
        {
            do_unary_operation(topology, op, activation_asin);
            break;
        }
        case OP_TYPEID::Sinh:
        {
            do_unary_operation(topology, op, activation_sinh);
            break;
        }
        case OP_TYPEID::Cos:
        {
            do_unary_operation(topology, op, activation_cos);
            break;
        }
        case OP_TYPEID::Acos:
        {
            do_unary_operation(topology, op, activation_acos);
            break;
        }
        case OP_TYPEID::Cosh:
        {
            do_unary_operation(topology, op, activation_cosh);
            break;
        }
        case OP_TYPEID::Log:
        {
            do_unary_operation(topology, op, activation_log);
            break;
        }
        case OP_TYPEID::Exp:
        {
            do_unary_operation(topology, op, activation_exp);
            break;
        }
        case OP_TYPEID::Sigmoid:
        {
            do_unary_operation(topology, op, activation_logistic);
            break;
        }
        case OP_TYPEID::SigmoidBackprop:
        {
            arguments_check(op, 2, 1);

            do_sigmoid_backprop_operation(topology,
                                          get_input_name(op, 0),
                                          get_input_shape(op, 0),
                                          get_input_name(op, 1),
                                          get_input_shape(op, 1),
                                          get_output_name(op),
                                          get_output_shape(op),
                                          get_output_type(op));
            break;
        }
        case OP_TYPEID::Not:
        {
            arguments_check(op, 1, 1);

            do_not_operation(topology,
                             get_input_name(op),
                             get_input_shape(op),
                             get_output_name(op),
                             get_output_shape(op),
                             get_output_type(op));
            break;
        }
        case OP_TYPEID::Greater:
        {
            do_logical_operation(topology, op, " > ");
            break;
        }
        case OP_TYPEID::GreaterEq:
        {
            do_logical_operation(topology, op, " >= ");
            break;
        }
        case OP_TYPEID::Equal:
        {
            do_logical_operation(topology, op, " == ");
            break;
        }
        case OP_TYPEID::NotEqual:
        {
            do_logical_operation(topology, op, " != ");
            break;
        }
        case OP_TYPEID::Less:
        {
            do_logical_operation(topology, op, " < ");
            break;
        }
        case OP_TYPEID::LessEq:
        {
            do_logical_operation(topology, op, " <= ");
            break;
        }
        case OP_TYPEID::And:
        {
            do_logical_operation(topology, op, " && ");
            break;
        }
        case OP_TYPEID::Or:
        {
            do_logical_operation(topology, op, " || ");
            break;
        }
        case OP_TYPEID::Subtract:
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::sub);
            break;
        }
        case OP_TYPEID::Power:
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::pow);
            break;
        }
        case OP_TYPEID::Atan:
        {
            arguments_check(op, 1, 1);
            do_custom_eltwise_operation(topology,
                                        get_input_name(op),
                                        get_input_shape(op),
                                        get_input_type(op),
                                        get_output_name(op),
                                        get_output_shape(op),
                                        get_output_type(op),
                                        CUSTOM_ELTWISE::Atan);
            break;
        }
        case OP_TYPEID::Ceiling:
        {
            arguments_check(op, 1, 1);
            do_custom_eltwise_operation(topology,
                                        get_input_name(op),
                                        get_input_shape(op),
                                        get_input_type(op),
                                        get_output_name(op),
                                        get_output_shape(op),
                                        get_output_type(op),
                                        CUSTOM_ELTWISE::Ceil);
            break;
        }
        case OP_TYPEID::Floor:
        {
            arguments_check(op, 1, 1);
            do_custom_eltwise_operation(topology,
                                        get_input_name(op),
                                        get_input_shape(op),
                                        get_input_type(op),
                                        get_output_name(op),
                                        get_output_shape(op),
                                        get_output_type(op),
                                        CUSTOM_ELTWISE::Floor);
            break;
        }
        case OP_TYPEID::Sign:
        {
            arguments_check(op, 1, 1);
            do_custom_eltwise_operation(topology,
                                        get_input_name(op),
                                        get_input_shape(op),
                                        get_input_type(op),
                                        get_output_name(op),
                                        get_output_shape(op),
                                        get_output_type(op),
                                        CUSTOM_ELTWISE::Sign);
            break;
        }
        case OP_TYPEID::Tan:
        {
            arguments_check(op, 1, 1);
            do_custom_eltwise_operation(topology,
                                        get_input_name(op),
                                        get_input_shape(op),
                                        get_input_type(op),
                                        get_output_name(op),
                                        get_output_shape(op),
                                        get_output_type(op),
                                        CUSTOM_ELTWISE::Tan);
            break;
        }
        case OP_TYPEID::Pad:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::Pad> pad = static_pointer_cast<op::Pad>(op);
            const Shape& pad_below = pad->get_padding_below();
            const Shape& pad_interior = pad->get_padding_interior();

            do_pad_operation(topology,
                             get_input_name(op, 0),
                             get_input_shape(op),
                             get_input_name(op, 1),
                             get_output_name(op),
                             get_output_shape(op),
                             get_output_type(op),
                             pad_below,
                             pad_interior);
            break;
        }
        case OP_TYPEID::BatchNormTrainingBackprop:
        {
            arguments_check(op, 6, 3);

            const shared_ptr<op::BatchNormTrainingBackprop> batch_norm =
                static_pointer_cast<op::BatchNormTrainingBackprop>(op);
            const double eps = batch_norm->get_eps_value();

            do_create_mean(topology,
                           get_output_name(op, 2), // d_beta
                           get_output_type(op, 2),
                           get_input_name(op, 5), // delta
                           get_input_shape(op, 5),
                           true);

            do_create_variance_back(topology,
                                    get_output_name(op, 1), // d_gamma
                                    get_output_type(op, 1),
                                    eps,
                                    get_input_name(op, 2), // input
                                    get_input_shape(op, 2),
                                    get_input_name(op, 3),  // gamma
                                    get_input_name(op, 4),  // beta
                                    get_input_name(op, 5)); // delta

            do_batch_norm_backprop_operation(topology,
                                             get_input_shape(op, 2),
                                             get_input_type(op, 2),
                                             get_input_name(op, 0),
                                             get_input_name(op, 1),
                                             get_input_name(op, 2),
                                             get_input_name(op, 3),
                                             get_input_name(op, 4),
                                             get_input_name(op, 5),
                                             eps,
                                             get_output_name(op, 0),
                                             get_output_name(op, 1),
                                             get_output_name(op, 2));
            break;
        }
        case OP_TYPEID::BatchNormInference:
        {
            const shared_ptr<op::BatchNormInference> bnorm =
                static_pointer_cast<op::BatchNormInference>(op);
            const double eps = bnorm->get_eps_value();

            arguments_check(op, 5, 1);

            if (get_input_name(op, 2).size() != 4)
            {
                do_batch_norm_operation(topology,
                                        get_output_name(op),
                                        get_output_type(op),
                                        eps,
                                        get_input_name(op, 2),
                                        get_input_shape(op, 2),
                                        get_input_name(op, 0),
                                        get_input_name(op, 1),
                                        get_input_name(op, 3),
                                        get_input_name(op, 4));
            }
            else
            {
                const cldnn::batch_norm batchnorm(get_output_name(op),
                                                  get_input_name(op, 2), // input
                                                  get_input_name(op, 3), // mean
                                                  get_input_name(op, 4), // variance
                                                  get_input_name(op, 0), // gamma
                                                  get_input_name(op, 1), // beta
                                                  eps);                  // epsilon (float)
                topology.add(batchnorm);
            }
            break;
        }
        case OP_TYPEID::BatchNormTraining:
        {
            const shared_ptr<op::BatchNormTraining> bnorm =
                static_pointer_cast<op::BatchNormTraining>(op);
            const double eps = bnorm->get_eps_value();

            if (get_input_name(op, 2).size() != 4)
            {
                string mean_name;
                string variance_name;

                if (op->get_inputs().size() < 3 || op->get_outputs().empty())
                {
                    arguments_check(op, 3, 1); // throw exception in this case
                }

                if (op->get_outputs().size() == 3)
                {
                    arguments_check(op, 3, 3);

                    mean_name = get_output_name(op, 1);
                    variance_name = get_output_name(op, 2);

                    do_create_mean(topology,
                                   mean_name,
                                   get_output_type(op),
                                   get_input_name(op, 2),
                                   get_input_shape(op, 2),
                                   false);

                    do_create_variance(topology,
                                       variance_name,
                                       get_output_type(op),
                                       get_input_name(op, 2),
                                       get_input_shape(op, 2),
                                       mean_name);
                }

                if (op->get_outputs().size() == 1 || op->get_outputs().size() == 3)
                {
                    if (mean_name.empty() || variance_name.empty())
                    {
                        arguments_check(op, 5, 1);

                        mean_name = get_input_name(op, 3);
                        variance_name = get_input_name(op, 4);
                    }

                    do_batch_norm_operation(topology,
                                            get_output_name(op),
                                            get_output_type(op),
                                            eps,
                                            get_input_name(op, 2),
                                            get_input_shape(op, 2),
                                            get_input_name(op, 0),
                                            get_input_name(op, 1),
                                            mean_name,
                                            variance_name);
                }
                else
                {
                    arguments_check(op, 5, 1); // throw exception in this case
                }
            }
            else
            {
                if (op->get_inputs().size() == 5 && op->get_outputs().size() == 1)
                {
                    const cldnn::batch_norm batchnorm(get_output_name(op),
                                                      get_input_name(op, 2), // input
                                                      get_input_name(op, 3), // mean
                                                      get_input_name(op, 4), // variance
                                                      get_input_name(op, 0), // gamma
                                                      get_input_name(op, 1), // beta
                                                      eps);                  // epsilon (float)
                    topology.add(batchnorm);
                }
                else if (op->get_inputs().size() == 3 && op->get_outputs().size() == 3)
                {
                    const string mean_name = get_output_name(op, 1);
                    const string variance_name = get_output_name(op, 2);

                    // Create a memory for mean as mutable_data to treat it as constant
                    const cldnn::layout mean_layout = IntelGPULayout::create_cldnn_layout(
                        get_output_type(op, 1), get_output_shape(op, 1));
                    const cldnn::memory mean_mem(cldnn::memory::allocate(*ocl_engine, mean_layout));

                    const cldnn::mutable_data mean_const(mean_name, mean_mem);
                    topology.add(mean_const);

                    // Create a memory for variance as mutable_data to treat it as constant
                    const cldnn::layout variance_layout = IntelGPULayout::create_cldnn_layout(
                        get_output_type(op, 2), get_output_shape(op, 2));
                    const cldnn::memory variance_mem(
                        cldnn::memory::allocate(*ocl_engine, variance_layout));

                    const cldnn::mutable_data variance_const(variance_name, variance_mem);
                    topology.add(variance_const);

                    const cldnn::batch_norm batchnorm(get_output_name(op),
                                                      get_input_name(op, 2), // input
                                                      eps,                   // epsilon (float)
                                                      mean_name,
                                                      variance_name,
                                                      get_input_name(op, 0),  // gamma
                                                      get_input_name(op, 1)); // beta
                    topology.add(batchnorm);

                    // Need to mark this operation as "output" to keep mean and variance
                    // in cldnn::network
                    func_output_names.insert(get_output_name(op));
                }
                else
                {
                    arguments_check(op, 5, 1); // throw exception in this case
                }
            }
            break;
        }
        case OP_TYPEID::Convolution:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::Convolution> conv_op = static_pointer_cast<op::Convolution>(op);
            const Strides& win_stride = conv_op->get_window_movement_strides();
            const Strides& win_dilation = conv_op->get_window_dilation_strides();
            const Strides& data_dilation = conv_op->get_data_dilation_strides();
            const CoordinateDiff& pad_below = conv_op->get_padding_below();
            const CoordinateDiff& pad_above = conv_op->get_padding_above();

            // clDNN has quite limited support for Convolution operation
            // following are the checks to go with workaround
            if ((win_stride.size() > 2) || (pad_below.size() > 2) || (pad_above.size() > 2) ||
                (win_dilation.size() > 2) || (data_dilation.size() > 2) ||
                (data_dilation.at(0) != 1) || (data_dilation.at(1) != 1))
            {
                do_convolution_operation(topology,
                                         get_input_name(op, 0),
                                         get_input_shape(op, 0),
                                         get_input_name(op, 1),
                                         get_input_shape(op, 1),
                                         get_output_name(op),
                                         get_output_shape(op),
                                         get_output_type(op),
                                         conv_op->get_padding_below(),
                                         conv_op->get_window_movement_strides(),
                                         conv_op->get_window_dilation_strides(),
                                         conv_op->get_data_dilation_strides(),
                                         0,
                                         1,
                                         1,
                                         "input[batch][input_channel]",
                                         "filter[output_channel][input_channel]",
                                         "output[batch][output_channel]",
                                         false);
            }
            else
            {
                cldnn::tensor::value_type input_offset_x = -pad_below.at(1);
                cldnn::tensor::value_type input_offset_y = -pad_below.at(0);
                std::string op_input_name = get_input_name(op, 0);

                if ((pad_below.at(0) != pad_above.at(0)) || (pad_below.at(1) != pad_above.at(1)))
                {
                    // Different input padding for operation workarounded by adding aux layer
                    const cldnn::tensor border_pad_above(0, 0, pad_below.at(1), pad_below.at(0));
                    const cldnn::tensor border_pad_below(0, 0, pad_above.at(1), pad_above.at(0));
                    input_offset_x = 0;
                    input_offset_y = 0;
                    op_input_name += "_bordered";

                    const cldnn::border cldnn_border(op_input_name,
                                                     get_input_name(op, 0),
                                                     border_pad_above,
                                                     border_pad_below,
                                                     cldnn::border_type::zero);
                    topology.add(cldnn_border);
                }

                const cldnn::tensor input_offset(0, 0, input_offset_x, input_offset_y);
                const cldnn::tensor strides(1, 1, win_stride.at(1), win_stride.at(0));
                const cldnn::tensor dilation(1, 1, win_dilation.at(1), win_dilation.at(0));

                const cldnn::convolution cldnn_conv(get_output_name(op),
                                                    op_input_name,
                                                    {get_input_name(op, 1)},
                                                    strides,
                                                    input_offset,
                                                    dilation);
                topology.add(cldnn_conv);
            }
            break;
        }
        case OP_TYPEID::ConvolutionBackpropFilters:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::ConvolutionBackpropFilters> conv_op =
                static_pointer_cast<op::ConvolutionBackpropFilters>(op);

            do_convolution_operation(topology,
                                     get_input_name(op, 0),
                                     get_input_shape(op, 0),
                                     get_input_name(op, 1),
                                     get_input_shape(op, 1),
                                     get_output_name(op),
                                     get_output_shape(op),
                                     get_output_type(op),
                                     conv_op->get_padding_below_backward(),
                                     conv_op->get_window_movement_strides_backward(),
                                     conv_op->get_window_dilation_strides_backward(),
                                     conv_op->get_data_dilation_strides_backward(),
                                     1,
                                     0,
                                     0,
                                     "input[input_channel][batch]",
                                     "filter[input_channel][output_channel]",
                                     "output[output_channel][batch]",
                                     false);
            break;
        }
        case OP_TYPEID::ConvolutionBackpropData:
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::ConvolutionBackpropData> conv_op =
                static_pointer_cast<op::ConvolutionBackpropData>(op);
            const Strides& win_stride = conv_op->get_window_movement_strides_backward();
            const CoordinateDiff& pad_below = conv_op->get_padding_below_backward();
            const CoordinateDiff& pad_above = conv_op->get_padding_above_backward();
            const Strides& win_dilation = conv_op->get_window_dilation_strides_backward();
            const Strides& data_dilation = conv_op->get_data_dilation_strides_backward();

            if ((win_stride.size() > 2) || (win_stride.at(0) != 1) || (win_stride.at(1) != 1) ||
                (pad_below.size() > 2) || (pad_above.size() > 2) || (data_dilation.size() > 2) ||
                (data_dilation.at(0) != 1) || (data_dilation.at(1) != 1) ||
                (win_dilation.size() > 2) || (win_dilation.at(0) != 1) || (win_dilation.at(1) != 1))
            {
                do_convolution_operation(topology,
                                         get_input_name(op, 1),
                                         get_input_shape(op, 1),
                                         get_input_name(op, 0),
                                         get_input_shape(op, 0),
                                         get_output_name(op),
                                         get_output_shape(op),
                                         get_output_type(op),
                                         conv_op->get_padding_below_backward(),
                                         conv_op->get_window_movement_strides_backward(),
                                         conv_op->get_window_dilation_strides_backward(),
                                         conv_op->get_data_dilation_strides_backward(),
                                         0,
                                         1,
                                         1,
                                         "input[batch][input_channel]",
                                         "filter[input_channel][output_channel]",
                                         "output[batch][output_channel]",
                                         true);
            }
            else
            {
                cldnn::tensor::value_type input_offset_xy = -1;
                std::string op_input_name = get_input_name(op, 1);

                if ((pad_below.at(0) == pad_above.at(0)) && (pad_below.at(1) == pad_above.at(1)))
                {
                    input_offset_xy = pad_below.at(0) - 1;
                }
                else
                {
                    // Different input padding for operation workarounded by adding aux layer
                    const cldnn::tensor crop_pad_above(0, 0, -pad_below.at(1), -pad_below.at(0));
                    const cldnn::tensor crop_pad_below(0, 0, -pad_above.at(1), -pad_above.at(0));
                    op_input_name += "_cropped";

                    const cldnn::crop cldnn_crop(op_input_name,
                                                 get_input_name(op, 1),
                                                 crop_pad_above,
                                                 crop_pad_below,
                                                 cldnn::crop_borders_t());
                    topology.add(cldnn_crop);
                }

                const cldnn::tensor input_offset(0, 0, input_offset_xy, input_offset_xy);
                const cldnn::tensor strides(1, 1, win_stride.at(1), win_stride.at(0));

                const cldnn::convolution_grad_input cldnn_conv_back_data(get_output_name(op),
                                                                         op_input_name,
                                                                         {get_input_name(op, 0)},
                                                                         strides,
                                                                         input_offset);
                topology.add(cldnn_conv_back_data);
            }
            break;
        }
        case OP_TYPEID::Min:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Min> min_op = static_pointer_cast<op::Min>(op);
            const AxisSet& axis = min_op->get_reduction_axes();

            do_max_min_operation(topology,
                                 get_input_name(op),
                                 get_input_shape(op),
                                 get_output_name(op),
                                 get_output_shape(op),
                                 get_output_type(op),
                                 axis,
                                 true);
            break;
        }
        case OP_TYPEID::Max:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Max> max_op = static_pointer_cast<op::Max>(op);
            const AxisSet& axis = max_op->get_reduction_axes();

            do_max_min_operation(topology,
                                 get_input_name(op),
                                 get_input_shape(op),
                                 get_output_name(op),
                                 get_output_shape(op),
                                 get_output_type(op),
                                 axis,
                                 false);
            break;
        }
        case OP_TYPEID::OneHot:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::OneHot> one_hot_op = static_pointer_cast<op::OneHot>(op);
            const size_t one_hot_axis = one_hot_op->get_one_hot_axis();

            do_one_hot_operation(topology,
                                 get_input_name(op),
                                 get_input_shape(op),
                                 get_input_type(op),
                                 get_output_name(op),
                                 get_output_shape(op),
                                 get_output_type(op),
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
                                         get_input_name(op),
                                         get_input_shape(op),
                                         get_input_type(op),
                                         get_output_name(op),
                                         get_output_shape(op),
                                         get_output_type(op),
                                         reduction_axis,
                                         true);
            }
            else
            {
                cldnn::arg_max_min::axis_name axis =
                    reduction_axis == 0 ? cldnn::arg_max_min::y : cldnn::arg_max_min::x;
                const cldnn::arg_max_min arg_max_min(
                    get_output_name(op), get_input_name(op), cldnn::arg_max_min::max, 1, axis);
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
                                         get_input_name(op),
                                         get_input_shape(op),
                                         get_input_type(op),
                                         get_output_name(op),
                                         get_output_shape(op),
                                         get_output_type(op),
                                         reduction_axis,
                                         false);
            }
            else
            {
                cldnn::arg_max_min::axis_name axis =
                    reduction_axis == 0 ? cldnn::arg_max_min::y : cldnn::arg_max_min::x;
                const cldnn::arg_max_min arg_max_min(
                    get_output_name(op), get_input_name(op), cldnn::arg_max_min::min, 1, axis);
                topology.add(arg_max_min);
            }
            break;
        }
        case OP_TYPEID::LRN:
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::LRN> lrn_op = static_pointer_cast<op::LRN>(op);

            const cldnn::lrn lrn(get_output_name(op),
                                 get_input_name(op),
                                 lrn_op->get_nsize(),
                                 lrn_op->get_bias(),
                                 lrn_op->get_alpha(),
                                 lrn_op->get_beta(),
                                 cldnn_lrn_norm_region_across_channel);
            topology.add(lrn);
            break;
        }
        case OP_TYPEID::AllReduce:
        case OP_TYPEID::BroadcastLike:
        case OP_TYPEID::FunctionCall:
        case OP_TYPEID::Dequantize:
        case OP_TYPEID::Quantize:
        case OP_TYPEID::ReduceWindow:
        case OP_TYPEID::ReplaceSlice:
        case OP_TYPEID::GenerateMask:
        case OP_TYPEID::ReverseSequence:
        case OP_TYPEID::ScalarConstantLike:
        case OP_TYPEID::SelectAndScatter:
        case OP_TYPEID::ShapeOf:
        case OP_TYPEID::StopGradient:
        case OP_TYPEID::TopK:
        case OP_TYPEID::EmbeddingLookup:
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

    instance.ocl_network =
        make_shared<cldnn::network>(*ocl_engine, topology, network_build_options);

    return func;
}

bool runtime::intelgpu::IntelGPUBackend::call(shared_ptr<Function> func,
                                              const vector<shared_ptr<runtime::Tensor>>& outputs,
                                              const vector<shared_ptr<runtime::Tensor>>& inputs)
{
    double mem_before_call = 0.0f;
    double mem_after_compilation = 0.0f;
    double mem_after_call = 0.0f;
    stopwatch timer_call;
    stopwatch timer_compile;

    if (m_profile_enable)
    {
        mem_before_call = get_max_memory_rss();
        timer_compile.start();
    }

    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network == nullptr)
    {
        throw runtime_error("compile() must be called before call().");
    }

    if (m_profile_enable)
    {
        timer_compile.stop();
        mem_after_compilation = get_max_memory_rss();
        timer_call.start();
    }

    shared_ptr<cldnn::network> network = instance.ocl_network;

    // Process input parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_parameters and inputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < inputs.size(); i++)
    {
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> tv =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(inputs[i]);
        const ParameterVector& input_params = func->get_parameters();
        const string& tensor_name = input_params[i]->get_output_tensor().get_name();
        network->set_input_data(tensor_name, *tv->get_data_ptr());
    }

    // Execute network
    map<cldnn::primitive_id, cldnn::network_output> result = network->execute();

    // Process output parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_results and outputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < func->get_output_size(); i++)
    {
        const shared_ptr<Node>& dst_node = func->get_output_op(i);
        const size_t dst_shape_size = shape_size(dst_node->get_shape());

        // We should not touch destination memory if it is not existed
        if (!dst_shape_size)
        {
            continue;
        }

        shared_ptr<runtime::intelgpu::IntelGPUTensorView> ngraph_res =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(outputs[i]);
        const string& tensor_name = get_input_name(dst_node);
        auto result_memory = result.at(tensor_name).get_memory().pointer<char>();

        memory_size_check(result_memory.size(), dst_node, func->get_name());

        ngraph_res->write(result_memory.data(), 0, result_memory.size());
    }

    if (m_profile_enable)
    {
        timer_call.stop();
        mem_after_call = get_max_memory_rss();

        print_call_performance(network,
                               func,
                               timer_compile.get_milliseconds(),
                               timer_call.get_milliseconds(),
                               mem_before_call,
                               mem_after_compilation,
                               mem_after_call);
    }

    if (m_function_cache_disabled)
    {
        remove_compiled_function(func);
    }

    return true;
}

void runtime::intelgpu::IntelGPUBackend::remove_compiled_function(shared_ptr<Function> func)
{
    ocl_networks.erase(func);
}

void runtime::intelgpu::IntelGPUBackend::enable_performance_data(shared_ptr<Function> func,
                                                                 bool enable)
{
    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network != nullptr)
    {
        throw runtime_error("Performance data collection must be enabled prior to compiling.");
    }

    instance.m_performance_counters_enabled = enable;
}

// The cldnn::network contains something like "generic_layer_0_Parameter_254_0" names
// This function should return "Parameter_254" from the example above
static string convert_cldnn_names(shared_ptr<Function> func, const string& cldnn_name)
{
    const string key("_");
    string result;

    const size_t last_key = cldnn_name.rfind(key);
    const size_t pre_last_key = cldnn_name.rfind(key, last_key - 1);
    const size_t pre_pre_last_key = cldnn_name.rfind(key, pre_last_key - 1);

    if (pre_pre_last_key == std::string::npos)
    {
        result = cldnn_name.substr(0, last_key);
    }
    else
    {
        result = cldnn_name.substr(pre_pre_last_key + 1, last_key - pre_pre_last_key - 1);
    }

    return result;
}

vector<runtime::PerformanceCounter>
    runtime::intelgpu::IntelGPUBackend::get_performance_data(shared_ptr<Function> func) const
{
    vector<runtime::PerformanceCounter> rc;
    auto it = ocl_networks.find(func);
    if (it != ocl_networks.end())
    {
        const shared_ptr<cldnn::network> network = it->second.ocl_network;

        if (network != nullptr && it->second.m_performance_counters_enabled)
        {
            const map<cldnn::primitive_id, cldnn::event>& primitives =
                network->get_executed_primitives();
            for (const auto& p : primitives)
            {
                // Let's generate the primitive name that matches to the name in Function
                const string primitive_name = convert_cldnn_names(func, p.first);
                size_t usec = 0;
                for (const auto& q : p.second.get_profiling_info())
                {
                    if (q.name == string("executing"))
                    {
                        usec += chrono::duration_cast<
                                    chrono::duration<size_t, chrono::milliseconds::period>>(
                                    q.value->value())
                                    .count();
                    }
                }
                const runtime::PerformanceCounter perf_counter(primitive_name.c_str(), usec, 1);
                rc.push_back(perf_counter);
            }
        }
    }
    return rc;
}

static Node* get_node_by_name(const shared_ptr<Function> func, const string& name)
{
    for (shared_ptr<Node> node : func->get_ops())
    {
        if (node->get_name() == name)
        {
            return node.get();
        }
    }

    return nullptr;
}

void runtime::intelgpu::IntelGPUBackend::print_call_performance(
    const shared_ptr<cldnn::network> network,
    const shared_ptr<Function> func,
    size_t time_compile,
    size_t time_call,
    double mem_before_call,
    double mem_after_compilation,
    double mem_after_call) const
{
    struct data_item
    {
        string item_name;
        map<string, double> item_times;
    };
    const string& func_name = func->get_name();
    const map<cldnn::primitive_id, cldnn::event>& primitives = network->get_executed_primitives();
    size_t limit_count = m_profile_lines_limit_count;
    multimap<double, data_item> data;
    map<string, double> total_interval_times;
    double total_executing_time = 0;
    size_t total_items_count = 0;
    size_t max_item_name_size = 0;

    ios_base::fmtflags saved_stream_flags(cout.flags()); // Save stream flags to restore them later

    if (m_profile_lines_limit_count > 0)
    {
        // Extract profiling statistic, calculate summary and sort
        for (auto& prim : primitives)
        {
            double executing_time = 0;
            data_item item;
            item.item_name = prim.first;
            max_item_name_size = max(max_item_name_size, prim.first.size());

            for (auto& prof_info : prim.second.get_profiling_info())
            {
                const string& interval_name = prof_info.name;
                double interval =
                    chrono::duration_cast<chrono::duration<double, chrono::milliseconds::period>>(
                        prof_info.value->value())
                        .count();

                item.item_times[interval_name] = interval;

                // Get the Key time to sort by
                if (interval_name == "executing")
                {
                    executing_time += interval;
                }

                // Accumulate total time for each interval
                if (total_interval_times.find(interval_name) == total_interval_times.end())
                {
                    total_interval_times[interval_name] = interval;
                }
                else
                {
                    total_interval_times[interval_name] += interval;
                }
            }
            data.emplace(executing_time, item);
            total_executing_time += executing_time;
            ++total_items_count;
        }

        // Print statistic for each primitive in the cldnn::network
        for (auto it = data.rbegin(); (it != data.rend()) && (limit_count > 0); ++it, --limit_count)
        {
            const string ngraph_node_name = convert_cldnn_names(func, it->second.item_name);
            const Node* ngraph_node = get_node_by_name(func, ngraph_node_name);

            cout << func_name << delim << setw(max_item_name_size) << it->second.item_name << delim
                 << "time(ms)" << delim << scientific << setprecision(2) << it->first;
            for (auto item : it->second.item_times)
            {
                cout << delim << item.first << "(ms)" << delim << item.second;
            }
            cout << delim << ngraph_node_name;

            if (ngraph_node) // it might be initialized by nullptr
            {
                // print all input shapes for the Node
                size_t arg_idx = 0;
                for (const descriptor::Input& op_input : ngraph_node->get_inputs())
                {
                    cout << delim << op_input.get_element_type().c_type_string() << " input"
                         << arg_idx << vector_to_string(op_input.get_shape());
                    ++arg_idx;
                }

                // print all output shapes for the Node
                arg_idx = 0;
                for (const descriptor::Output& op_output : ngraph_node->get_outputs())
                {
                    cout << delim << op_output.get_element_type().c_type_string() << " output"
                         << arg_idx << vector_to_string(op_output.get_shape());
                    ++arg_idx;
                }
            }

            cout << "\n";
        }

        // Print bottom line summary
        const string total_items_count_string = "Total(cldnn " + to_string(total_items_count) +
                                                ", ngraph " + to_string(func->get_ops().size()) +
                                                ")";
        cout << func_name << delim << setw(max_item_name_size) << total_items_count_string << delim
             << "time(ms)" << delim << scientific << setprecision(2) << total_executing_time;
        for (auto item_times : total_interval_times)
        {
            cout << delim << item_times.first << "(ms)" << delim << item_times.second;
        }
        cout << "\n";
    }

    // Print time and memory consumed in ::call function
    cout << func_name << delim << " Backend compilation(ms)" << delim << time_compile << " call(ms)"
         << delim << time_call << delim << "memory before call(B)" << delim << mem_before_call
         << delim << "after compilation(B)" << delim << mem_after_compilation << delim
         << "after call(B)" << delim << mem_after_call << endl;

    cout.flags(saved_stream_flags); // Restore stream configuration to leave it in original state
}
