/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <CPP/activation.hpp>
#include <CPP/activation_grad.hpp>
#include <CPP/batch_norm.hpp>
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
#include <CPP/topology.hpp>

#include "ngraph/runtime/intelgpu/intelgpu_backend.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_layout.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_batchnorm.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_broadcast.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_convolution.hpp"
#include "ngraph/runtime/intelgpu/intelgpu_op_custom_kernels.hpp"
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
#include "ngraph/op/pad.hpp"
#include "ngraph/op/parameter_vector.hpp"
#include "ngraph/op/product.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/reverse.hpp"
#include "ngraph/op/slice.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

static void arguments_check(const shared_ptr<Node>& op, size_t input, size_t output)
{
    if (op->get_input_size() != input || op->get_output_size() != output)
    {
        ostringstream os;
        os << "Operation \"" << op->description() << "\" input and output sizes mismatch."
           << " Expected input size=" << op->get_input_size() << ", provided=" << input
           << ". Expected output size=" << op->get_output_size() << ", provided=" << output;
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

static void argument_type_check(const element::Type& type)
{
    if (type != element::f32 && type != element::boolean)
    {
        throw invalid_argument("Kernel data type \"" + type.c_type_string() +
                               "\" is not supported.");
    }
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
                                 const Shape& pad_above,
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
    argument_type_check(get_input_type(op, 0));
    argument_type_check(get_input_type(op, 1));

    runtime::intelgpu::do_logic_kernel(topology,
                                       get_input_name(op, 0),
                                       get_input_shape(op, 0),
                                       get_input_type(op, 0).c_type_string(),
                                       get_input_name(op, 1),
                                       get_input_shape(op, 1),
                                       get_input_type(op, 1).c_type_string(),
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
        if ("Parameter" == op->description())
        {
            arguments_check(op, 0, 1);

            const string& element_name = op->get_output_tensor_view()->get_tensor().get_name();
            const cldnn::layout element_layout =
                IntelGPULayout::create_cldnn_layout(op->get_element_type(), op->get_shape());

            const cldnn::input_layout op_layout(element_name, element_layout);
            topology.add(op_layout);
        }
        else if ("Result" == op->description())
        {
            arguments_check(op, 1, 1);

            do_equal_propagation(topology, get_input_name(op), get_output_name(op));
        }
        else if ("GetOutputElement" == op->description())
        {
            arguments_check(op, 3, 1);

            const shared_ptr<op::GetOutputElement> elem =
                static_pointer_cast<op::GetOutputElement>(op);

            do_equal_propagation(topology, get_input_name(op, elem->get_n()), get_output_name(op));
        }
        else if ("Slice" == op->description())
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
        }
        else if ("Select" == op->description())
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
        }
        else if ("Reverse" == op->description())
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
        }
        else if ("Convert" == op->description())
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
        }
        else if ("Concat" == op->description())
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
        }
        else if ("Add" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::sum);
        }
        else if ("Multiply" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::prod);
        }
        else if ("Divide" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::div);
        }
        else if ("Maximum" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::max);
        }
        else if ("Constant" == op->description())
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
        }
        else if ("Dot" == op->description())
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
        }
        else if ("MaxPool" == op->description())
        {
            const shared_ptr<op::MaxPool> max_pool = static_pointer_cast<op::MaxPool>(op);
            const Shape& pool_shape = max_pool->get_window_shape();
            const Strides& pool_strides = max_pool->get_window_movement_strides();
            const Shape& pad_below = max_pool->get_padding_below();
            const Shape& pad_above = max_pool->get_padding_above();

            do_pooling_operation(topology,
                                 op,
                                 pool_shape,
                                 pool_strides,
                                 pad_below,
                                 pad_above,
                                 cldnn::pooling_mode::max);
        }
        else if ("AvgPool" == op->description())
        {
            const shared_ptr<op::AvgPool> avg_pool = static_pointer_cast<op::AvgPool>(op);
            const Shape& pool_shape = avg_pool->get_window_shape();
            const Strides& pool_strides = avg_pool->get_window_movement_strides();
            const Shape& pad_below = avg_pool->get_padding_below();
            const Shape& pad_above = avg_pool->get_padding_above();
            const cldnn::pooling_mode mode = avg_pool->get_include_padding_in_avg_computation()
                                                 ? cldnn::pooling_mode::average
                                                 : cldnn::pooling_mode::average_no_padding;

            do_pooling_operation(
                topology, op, pool_shape, pool_strides, pad_below, pad_above, mode);
        }
        else if ("Broadcast" == op->description())
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Broadcast> broadcast = static_pointer_cast<op::Broadcast>(op);
            const AxisSet& axis = broadcast->get_broadcast_axes();

            if (axis.empty())
            {
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else if (get_input_shape(op).empty())
            {
                do_bcast_sum_operation_scalar(topology,
                                              get_input_name(op),
                                              get_input_shape(op),
                                              get_output_name(op),
                                              get_output_shape(op),
                                              get_output_type(op),
                                              true);
            }
            else
            {
                do_bcast_sum_operation(topology,
                                       get_input_name(op),
                                       get_input_shape(op),
                                       get_output_name(op),
                                       get_output_shape(op),
                                       get_output_type(op),
                                       axis,
                                       true);
            }
        }
        else if ("Sum" == op->description())
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Sum> sum = static_pointer_cast<op::Sum>(op);
            const AxisSet& axis = sum->get_reduction_axes();

            if (axis.empty())
            {
                do_equal_propagation(topology, get_input_name(op), get_output_name(op));
            }
            else if (get_output_shape(op).empty())
            {
                do_bcast_sum_operation_scalar(topology,
                                              get_input_name(op),
                                              get_input_shape(op),
                                              get_output_name(op),
                                              get_output_shape(op),
                                              get_output_type(op),
                                              false);
            }
            else
            {
                do_bcast_sum_operation(topology,
                                       get_input_name(op),
                                       get_input_shape(op),
                                       get_output_name(op),
                                       get_output_shape(op),
                                       get_output_type(op),
                                       axis,
                                       false);
            }
        }
        else if ("Product" == op->description())
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
        }
        else if ("Reshape" == op->description())
        {
            arguments_check(op, 1, 1);

            const shared_ptr<op::Reshape> op_broadcast = static_pointer_cast<op::Reshape>(op);
            const AxisVector& broadcast_axes = op_broadcast->get_input_order();

            vector<uint16_t> permute_order({0, 1, 2, 3}); // No action by default
            const size_t max_dim = 4;
            const size_t scale =
                broadcast_axes.size() < max_dim ? max_dim - broadcast_axes.size() : 0;

            // Need to scale indexes up according on array rank.
            // For example, in 2D array, indexes are 0,1 but in 4D array it should be 2,3
            // because cldnn::tensor is always 4D assuming cldnn::bfyx model
            size_t rindex = max_dim;
            for (auto i = broadcast_axes.crbegin(); i != broadcast_axes.crend() && rindex > 0;
                 ++i, --rindex)
            {
                permute_order.at(rindex - 1) = *i + scale;
            }

            const cldnn::permute cldnn_permute(
                get_output_name(op), get_input_name(op), permute_order);
            topology.add(cldnn_permute);
        }
        else if ("Negative" == op->description())
        {
            const cldnn_activation_additional_params param = {-1.f, 0.f};
            do_unary_operation(topology, op, activation_linear, param);
        }
        else if ("Relu" == op->description())
        {
            do_unary_operation(topology, op, activation_relu);
        }
        else if ("ReluBackprop" == op->description())
        {
            arguments_check(op, 2, 1);

            const cldnn_activation_additional_params& param = {0.f, 0.f};
            const cldnn::activation_grad cldnn_activ_grad(get_output_name(op),
                                                          get_input_name(op, 1),
                                                          get_input_name(op, 0),
                                                          activation_grad_relu,
                                                          param);
            topology.add(cldnn_activ_grad);
        }
        else if ("Abs" == op->description())
        {
            do_unary_operation(topology, op, activation_abs);
        }
        else if ("Sqrt" == op->description())
        {
            do_unary_operation(topology, op, activation_sqrt);
        }
        else if ("Tanh" == op->description())
        {
            do_unary_operation(topology, op, activation_hyperbolic_tan);
        }
        else if ("Sin" == op->description())
        {
            do_unary_operation(topology, op, activation_sin);
        }
        else if ("Asin" == op->description())
        {
            do_unary_operation(topology, op, activation_asin);
        }
        else if ("Sinh" == op->description())
        {
            do_unary_operation(topology, op, activation_sinh);
        }
        else if ("Cos" == op->description())
        {
            do_unary_operation(topology, op, activation_cos);
        }
        else if ("Acos" == op->description())
        {
            do_unary_operation(topology, op, activation_acos);
        }
        else if ("Cosh" == op->description())
        {
            do_unary_operation(topology, op, activation_cosh);
        }
        else if ("Log" == op->description())
        {
            do_unary_operation(topology, op, activation_log);
        }
        else if ("Exp" == op->description())
        {
            do_unary_operation(topology, op, activation_exp);
        }
        else if ("Sigmoid" == op->description())
        {
            do_unary_operation(topology, op, activation_logistic);
        }
        else if ("Greater" == op->description())
        {
            do_logical_operation(topology, op, " > ");
        }
        else if ("GreaterEq" == op->description())
        {
            do_logical_operation(topology, op, " >= ");
        }
        else if ("Equal" == op->description())
        {
            do_logical_operation(topology, op, " == ");
        }
        else if ("NotEqual" == op->description())
        {
            do_logical_operation(topology, op, " != ");
        }
        else if ("Less" == op->description())
        {
            do_logical_operation(topology, op, " < ");
        }
        else if ("LessEq" == op->description())
        {
            do_logical_operation(topology, op, " <= ");
        }
        else if ("And" == op->description())
        {
            do_logical_operation(topology, op, " && ");
        }
        else if ("Or" == op->description())
        {
            do_logical_operation(topology, op, " || ");
        }
        else if ("Subtract" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::sub);
        }
        else if ("Power" == op->description())
        {
            do_eltwise_operation(topology, op, cldnn::eltwise_mode::pow);
        }
        else if ("Pad" == op->description())
        {
            arguments_check(op, 2, 1);

            const shared_ptr<op::Pad> pad = static_pointer_cast<op::Pad>(op);
            const Shape& pad_above = pad->get_padding_above();
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
        }
        else if ("BatchNorm" == op->description())
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
                               get_input_shape(op),
                               get_output_type(op),
                               get_input_name(op, 2),
                               get_input_shape(op, 2));

                do_create_variance(topology,
                                   variance_name,
                                   get_input_shape(op),
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
                                        get_output_shape(op),
                                        get_output_type(op),
                                        eps,
                                        get_input_name(op, 2),
                                        get_input_shape(op, 2),
                                        get_input_name(op, 0),
                                        get_input_shape(op, 0),
                                        get_input_name(op, 1),
                                        mean_name,
                                        variance_name);
            }
            else
            {
                arguments_check(op, 5, 1); // throw exception in this case
            }
        }
        else if ("Convolution" == op->description())
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
        }
        else if ("ConvolutionBackpropFilters" == op->description())
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
        }
        else if ("ConvolutionBackpropData" == op->description())
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
        }
        else if ("Min" == op->description())
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
        }
        else if ("Max" == op->description())
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
        }
        else
        {
            throw invalid_argument("IntelGPU: Unsupported operation \"" + op->description() + "\"");
        }
    }

    instance.ocl_network = make_shared<cldnn::network>(*ocl_engine, topology);

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
