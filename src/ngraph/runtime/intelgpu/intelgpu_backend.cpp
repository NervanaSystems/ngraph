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

#include <CPP/activation.hpp>
#include <CPP/activation_grad.hpp>
#include <CPP/batch_norm.hpp>
#include <CPP/broadcast.hpp>
#include <CPP/concatenation.hpp>
#include <CPP/convolution.hpp>
#include <CPP/data.hpp>
#include <CPP/eltwise.hpp>
#include <CPP/input_layout.hpp>
#include <CPP/layout.hpp>
#include <CPP/permute.hpp>
#include <CPP/pooling.hpp>
#include <CPP/reorder.hpp>
#include <CPP/reshape.hpp>
#include <CPP/scale.hpp>
#include <CPP/softmax.hpp>
#include <CPP/topology.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_batchnorm.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_broadcast.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_convolution.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_softmax.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_tensor_view.hpp"

#include "ngraph/function.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/avg_pool.hpp"
#include "ngraph/op/batch_norm.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/convolution.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/get_output_element.hpp"
#include "ngraph/op/max.hpp"
#include "ngraph/op/max_pool.hpp"
#include "ngraph/op/min.hpp"
#include "ngraph/op/one_hot.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter_vector.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/softmax.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

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

    const cldnn::eltwise op_add(
        get_output_name(op), {get_input_name(op, 0), get_input_name(op, 1)}, mode);
    topology.add(op_add);
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

    const cldnn::tensor output_size =
        runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(get_output_shape(op));

    const cldnn::tensor input_offset =
        runtime::intelgpu::IntelGPULayout::create_cldnn_offset(pad_below);
    const cldnn::tensor size = runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(pool_shape);
    const cldnn::tensor stride =
        runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(pool_strides);

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

runtime::intelgpu::IntelGPUBackend::IntelGPUBackend()
{
    ocl_engine = make_shared<cldnn::engine>();
}

shared_ptr<runtime::TensorView>
    runtime::intelgpu::IntelGPUBackend::create_tensor(const element::Type& element_type,
                                                      const Shape& shape)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(element_type, shape, *ocl_engine);
}

shared_ptr<runtime::TensorView> runtime::intelgpu::IntelGPUBackend::create_tensor(
    const element::Type& element_type, const Shape& shape, void* memory_pointer)
{
    return make_shared<runtime::intelgpu::IntelGPUTensorView>(
        element_type, shape, *ocl_engine, memory_pointer);
}

bool runtime::intelgpu::IntelGPUBackend::compile(shared_ptr<Function> func)
{
    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network != nullptr)
    {
        return true;
    }

    cldnn::topology topology;

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

            do_equal_propagation(topology, get_input_name(op), get_output_name(op));
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
            const size_t ngraph_tensor_dims = get_input_shape(op, 0).size();
            const shared_ptr<op::Concat> concat_op = static_pointer_cast<op::Concat>(op);
            const size_t ngraph_concat_axis = concat_op->get_concatenation_axis();
            vector<cldnn::primitive_id> inputs;

            cldnn::concatenation::concatenation_axis cldnn_axis =
                runtime::intelgpu::IntelGPULayout::get_cldnn_axis(ngraph_tensor_dims,
                                                                  ngraph_concat_axis);

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

            do_dot_operation(topology,
                             get_input_name(op, 0),
                             get_input_shape(op, 0),
                             get_input_name(op, 1),
                             get_input_shape(op, 1),
                             get_output_name(op),
                             get_output_shape(op),
                             get_output_type(op));
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
            arguments_check(op, 2, 1);

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
            else if (get_input_shape(op).empty() ||
                     (get_input_shape(op).size() == 1 && get_input_shape(op).at(0) == 1))
            {
                const cldnn::tensor output_tensor_size =
                    runtime::intelgpu::IntelGPULayout::create_cldnn_tensor(get_output_shape(op));
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
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            break;
        }
        case OP_TYPEID::Negative:
        {
            const cldnn_activation_additional_params param = {-1.f, 0.f};
            do_unary_operation(topology, op, activation_linear, param);
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
        case OP_TYPEID::BatchNormBackprop:
        {
            arguments_check(op, 6, 3);

            const shared_ptr<op::BatchNormBackprop> batch_norm =
                static_pointer_cast<op::BatchNormBackprop>(op);
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
        case OP_TYPEID::BatchNorm:
        {
            const shared_ptr<op::BatchNorm> batch_norm = static_pointer_cast<op::BatchNorm>(op);
            const double eps = batch_norm->get_eps_value();
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
            if ((win_stride.size() > 2) || (pad_below.size() > 2 || pad_above.size() > 2) ||
                (pad_below.at(0) != pad_above.at(0) || pad_below.at(1) != pad_above.at(1)) ||
                (win_dilation.size() > 2) ||
                (data_dilation.size() > 2 || data_dilation.at(0) != 1 || data_dilation.at(1) != 1))
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
                const cldnn::tensor input_offset(0, 0, -pad_below.at(1), -pad_below.at(0));
                const cldnn::tensor strides(1, 1, win_stride.at(1), win_stride.at(0));
                const cldnn::tensor dilation(1, 1, win_dilation.at(1), win_dilation.at(0));

                const cldnn::convolution cldnn_conv(get_output_name(op),
                                                    get_input_name(op, 0),
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
        case OP_TYPEID::AllReduce:
        case OP_TYPEID::ArgMax:
        case OP_TYPEID::ArgMin:
        case OP_TYPEID::FunctionCall:
        case OP_TYPEID::LRN:
        case OP_TYPEID::Reduce:
        case OP_TYPEID::ReduceWindow:
        case OP_TYPEID::ReplaceSlice:
        case OP_TYPEID::ReverseSequence:
        case OP_TYPEID::SelectAndScatter:
        case OP_TYPEID::StopGradient:
        case OP_TYPEID::TopK:
        {
            throw unsupported_op("Unsupported op '" + op->description() +
                                 "' in IntelGPU back end.");
        }
#pragma GCC diagnostic pop
        }
    }

    cldnn::build_options network_build_options(cldnn::build_option::optimize_data(true));

    instance.ocl_network =
        make_shared<cldnn::network>(*ocl_engine, topology, network_build_options);

    return true;
}

bool runtime::intelgpu::IntelGPUBackend::call(
    shared_ptr<Function> func,
    const vector<shared_ptr<runtime::TensorView>>& outputs,
    const vector<shared_ptr<runtime::TensorView>>& inputs)
{
    validate_call(func, outputs, inputs);

    FunctionInstance& instance = ocl_networks[func];
    if (instance.ocl_network == nullptr)
    {
        if (!compile(func))
        {
            return false;
        }
    }

    shared_ptr<cldnn::network> network = instance.ocl_network;

    // Process input parameters. Correctness of parameters was validated by validate_call.
    // Since we have no correlation between Function::m_parameters and inputs, there is
    // we try to match them by index number in vectors.
    for (size_t i = 0; i < inputs.size(); i++)
    {
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> tv =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(inputs[i]);
        const op::ParameterVector& input_params = func->get_parameters();
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
        shared_ptr<runtime::intelgpu::IntelGPUTensorView> ngraph_res =
            static_pointer_cast<runtime::intelgpu::IntelGPUTensorView>(outputs[i]);
        const string& tensor_name = func->get_output_op(i)->get_output_tensor().get_name();

        auto result_memory = result.at(tensor_name).get_memory().pointer<char>();
        ngraph_res->write(result_memory.data(), 0, result_memory.size());
    }

    return true;
}
